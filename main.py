import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import random
import numpy as np
from src.models.evflownet import EVFlowNet
from src.datasets import DatasetProvider
from enum import Enum, auto
from src.datasets import train_collate
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
import os
import time

from image_preprocessing import combined_transform
from models.pclnet import PCLNet
from losses import TotalLoss

device = 'cpu'

class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def compute_epe_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor):
    '''
    end-point-error (ground truthと予測値の二乗誤差)を計算
    pred_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 予測したオプティカルフローデータ
    gt_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 正解のオプティカルフローデータ
    '''
    epe = torch.mean(torch.mean(torch.norm(pred_flow - gt_flow, p=2, dim=1), dim=(1, 2)), dim=0)
    return epe

def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    '''
    optical flowをnpyファイルに保存
    flow: torch.Tensor, Shape: torch.Size([2, 480, 640]) => オプティカルフローデータ
    file_name: str => ファイル名
    '''
    np.save(f"{file_name}.npy", flow.cpu().numpy())

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    set_seed(args.seed)
    '''
        ディレクトリ構造:

        data
        ├─test
        |  ├─test_city
        |  |    ├─events_left
        |  |    |   ├─events.h5
        |  |    |   └─rectify_map.h5
        |  |    └─forward_timestamps.txt
        └─train
            ├─zurich_city_11_a
            |    ├─events_left
            |    |       ├─ events.h5
            |    |       └─ rectify_map.h5
            |    ├─ flow_forward
            |    |       ├─ 000134.png
            |    |       |.....
            |    └─ forward_timestamps.txt
            ├─zurich_city_11_b
            └─zurich_city_11_c
        '''

    # ------------------
    #    Dataloader
    # ------------------

    loader = DatasetProvider(
        dataset_path=Path(args.dataset_path),
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        num_bins=4,
        transforms=combined_transform() # Custom class
    )
    train_set = loader.get_train_dataset()
    test_set = loader.get_test_dataset()

    # def split_train_valid(dataset):
    #     train_indices = []
    #     valid_indices = []
    #     for idx in range(len(dataset)):
    #         sample = dataset[idx]
    #         if 'flow_gt_valid_mask' in sample and sample['flow_gt_valid_mask'].all():
    #             valid_indices.append(idx)
    #         else:
    #             train_indices.append(idx)
    #     train_subset = torch.utils.data.Subset(dataset, train_indices)
    #     valid_subset = torch.utils.data.Subset(dataset, valid_indices)
    #     return train_subset, valid_subset

    # train_set_split, valid_set_split = split_train_valid(train_set)

    collate_fn = train_collate
    train_data = DataLoader(train_set, # train_set_split
                            batch_size=args.batch_size, #
                            shuffle=False,
                            collate_fn=collate_fn,
                            drop_last=False,
                            num_workers=os.cpu_count(),
                            pin_memory=True)
    test_data = DataLoader(test_set,
                        batch_size=1,
                        shuffle=False,
                        collate_fn=collate_fn,
                        drop_last=False,
                        num_workers=os.cpu_count(),
                        pin_memory=True)

    '''
    train data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
        Key: flow_gt, Type: torch.Tensor, Shape: torch.Size([Batch, 2, 480, 640]) => オプティカルフローデータのバッチ
        Key: flow_gt_valid_mask, Type: torch.Tensor, Shape: torch.Size([Batch, 1, 480, 640]) => オプティカルフローデータのvalid. ベースラインでは使わない

    test data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
    '''
    # ------------------
    #       Model
    # ------------------
    # model = EVFlowNet(args.train).to(device)
    model = torch.nn.DataParallel(PCLNet(args).to(device))

    # ------------------
    #   optimizer
    # ------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00002, weight_decay=args.wdecay, eps=args.epsilon)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-5, args.num_steps + 100,
                                                    pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    loss_fn = TotalLoss(smoothness_weight=0.5)

    # num_epochs = args.train.epochs
    num_epochs = 1

    epe_losses = [[] for _ in range(num_epochs)]
    overall_losses = [[] for _ in range(num_epochs)]

    WINDOW_FACTOR = 2
    WINDOW = args.snippet_len * WINDOW_FACTOR

    def save_model(model, additional_string: str = None):
        current_time = time.strftime("%Y%m%d-%H%M%S")
        model_path = f"models/model_{current_time}"
        if additional_string:
            model_path += '_' + additional_string
        model_path += '.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # ------------------
    #   Start training
    # ------------------

    model.train()

    for epoch in range(num_epochs):

        total_loss = 0
        prev_event_volumes = [torch.zeros([args.batch_size, 4, 480, 640])] * WINDOW # Acts as a queue

        print("on epoch: {}".format(epoch + 1))
        for i, batch in enumerate(tqdm(train_data)):

            try:
                batch: Dict[str, Any]

                event_image = batch["event_volume"].to(device) # [B, 3, 480, 640]
                ground_truth_flow = batch["flow_gt"].to(device) # [B, 2, 480, 640]

                prev_event_volumes.append(event_image)

                # input_tensor1 = [v.to(device) for v in prev_event_volumes[-WINDOW:]]
                input_tensor1 = prev_event_volumes[-WINDOW:]
                input_tensor = torch.stack([
                    torch.mean(torch.stack(input_tensor1[b: b + WINDOW_FACTOR], dim=0), dim=0)
                    for b in range(0, WINDOW, WINDOW_FACTOR) ], dim=1)
                _, flows = model(input_tensor) # [B, 3, 480, 640]

                # Overall loss requires flow0, ..., flow3 so we don't implement it here
                # What if you created flow_dict from flow0, ..., flow11 (n=12) to and then use overall loss?

                for j, flow in enumerate(flows):
                    print(f'batch {i} | flow #{j + 1} | EPE LOSS:', compute_epe_error(flow, ground_truth_flow).item())

                avg_flow = torch.mean(torch.stack(flows, dim=0), dim=0)
                epe_loss: torch.Tensor = compute_epe_error(avg_flow, ground_truth_flow)

                print(f"batch {i} average EPE LOSS: {epe_loss.item()}")
                epe_losses[epoch].append(epe_loss.item())

    #             print(f'batch {i} OVERALL LOSS: {loss_fn()}')

                optimizer.zero_grad()

                epe_loss.backward() # Change this to which loss function is to be updated
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                # xm.optimizer_step(optimizer)
                # xm.mark_step()

                total_loss += epe_loss.item() # This too

                if len(prev_event_volumes) >= WINDOW:
                    prev_event_volumes.pop(0) # Remove first element

                if (i + 1) % 10 == 0:
                    save_model(model, f'batch{i + 1}')

            except KeyboardInterrupt:
                save_model(model)
                continue
                # raise SystemExit("KeyboardInterrupt")

        scheduler.step()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_data)}')

    import time
    # Create the directory if it doesn't exist
    # if not os.path.exists('checkpoints'):
    #     os.makedirs('checkpoints')

    current_time = time.strftime("%Y%m%d-%H%M%S")
    model_path = f"models/pclnet_model_{current_time}_epoch1.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    model.eval()
    flow: torch.Tensor = torch.tensor([]).to(device)

    prev_event_volumes = [torch.zeros([1, 4, 480, 640])] * WINDOW # Acts as a queue

    with torch.no_grad():
        print("start test")
        for batch in tqdm(test_data):
            batch: Dict[str, Any]

            event_image = batch["event_volume"].to(device)
            prev_event_volumes.append(event_image)

            input_tensor1 = prev_event_volumes[-WINDOW:]
            input_tensor = torch.stack([
                torch.mean(torch.stack(input_tensor1[b: b + WINDOW_FACTOR], dim=0), dim=0)
                for b in range(0, WINDOW, WINDOW_FACTOR) ], dim=1)
            _, flows = model(input_tensor)

            batch_flow = torch.mean(torch.stack(flows, dim=0), dim=0)
            flow = torch.cat((flow, batch_flow), dim=0)  # [N, 2, 480, 640]

            if len(prev_event_volumes) >= WINDOW:
                prev_event_volumes.pop(0)

        print("test done")

    # ------------------
    #  save submission
    # ------------------
    current_time = time.strftime("%Y%m%d-%H%M%S")
    save_optical_flow_to_npy(flow, f'submissions/submission_pclnet_{current_time}')

if __name__ == "__main__":
    main()
