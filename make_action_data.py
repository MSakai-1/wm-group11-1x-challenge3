import numpy as np
import os
import json

# メタデータとバイナリファイルのパス設定
metadata_path = "data/train_v1.1/metadata.json"
base_path = "data/train_v1.1/actions/"
output_action_data_dir = "action_data"  # 保存先ディレクトリ（全データ）
output_normalized_changes_dir = "normalized_data" 

# 保存先ディレクトリ作成
os.makedirs(output_action_data_dir, exist_ok=True)
os.makedirs(output_normalized_changes_dir, exist_ok=True)

# メタデータを読み込む
with open(metadata_path, "r") as f:
    metadata = json.load(f)

num_images = metadata["num_images"]  # 総フレーム数
num_joints = 21  # 関節数
dtype = np.float32  # データ型

### 1. 関節データを処理して保存
joint_pos_path = os.path.join(base_path, "joint_pos.bin")
joint_data = np.memmap(joint_pos_path, dtype=dtype, mode="r", shape=(num_images, num_joints))

# 各関節ごとに保存
for joint_idx in range(num_joints):
    joint_all_frames = joint_data[:, joint_idx]
    joint_output_path = os.path.join(output_action_data_dir, f"joint_{joint_idx:02d}.npy")
    np.save(joint_output_path, joint_all_frames)

    # 変化データの保存
    joint_changes = np.zeros_like(joint_all_frames, dtype=np.int8)
    joint_changes[1:] = np.sign(joint_all_frames[1:] - joint_all_frames[:-1])
    change_output_path = os.path.join(output_action_data_dir, f"joint_{joint_idx:02d}_changes.npy")
    np.save(change_output_path, joint_changes)

for file_name in sorted(os.listdir(output_action_data_dir)):
    if file_name.endswith("_changes.npy"):
        # データを読み込む
        file_path = os.path.join(output_action_data_dir, file_name)
        changes = np.load(file_path)  # 形状: (num_frames,)

        # 正規化
        max_abs_value = np.max(np.abs(changes))  # 各関節の変化の絶対値の最大値
        if max_abs_value > 0:
            normalized_changes = changes / max_abs_value
        else:
            normalized_changes = changes  # 最大値が0の場合、データをそのまま保持

        # 出力ファイル名と保存
        normalized_file_name = file_name.replace("_changes.npy", "_changes_normalized.npy")
        output_file_path = os.path.join(output_normalized_changes_dir, normalized_file_name)
        np.save(output_file_path, normalized_changes)
        print(f"Normalized changes for {file_name} saved to {output_file_path}")

### 2. l_hand_closure と r_hand_closure のデータ保存
for hand in ["l_hand_closure", "r_hand_closure"]:
    hand_path = os.path.join(base_path, f"{hand}.bin")
    hand_data = np.memmap(hand_path, dtype=dtype, mode="r", shape=(num_images,))
    hand_output_path = os.path.join(output_action_data_dir, f"{hand}.npy")
    np.save(hand_output_path, hand_data)

### 3. driving_command の速度と角速度を処理
driving_command_path = os.path.join(base_path, "driving_command.bin")
driving_command_data = np.memmap(driving_command_path, dtype=dtype, mode="r", shape=(num_images, 2))

# 速度と角速度を保存
velocity_data = driving_command_data[:, 0]
angular_velocity_data = driving_command_data[:, 1]

np.save(os.path.join(output_action_data_dir, "velocity.npy"), velocity_data)
np.save(os.path.join(output_action_data_dir, "angular_velocity.npy"), angular_velocity_data)

# 速度加速度の正規化データの計算と保存
velocity_normalized = np.zeros_like(velocity_data, dtype=np.int8)
velocity_normalized[1:] = np.sign(velocity_data[1:] - velocity_data[:-1])
angular_velocity_normalized = np.zeros_like(angular_velocity_data, dtype=np.int8)
angular_velocity_normalized[1:] = np.sign(angular_velocity_data[1:] - angular_velocity_data[:-1])

np.save(os.path.join(output_normalized_changes_dir, "velocity_normalized.npy"), velocity_normalized)
np.save(os.path.join(output_normalized_changes_dir, "angular_velocity_normalized.npy"), angular_velocity_normalized)

### 4. 各フレームの action データをまとめる
action_data_combined_dir = "combined_action_data"
os.makedirs(action_data_combined_dir, exist_ok=True)

combined_data_normalized_dir = "combined_data_normalized"
os.makedirs(combined_data_normalized_dir, exist_ok=True)

#正規化前のデータセット作成
#joint_changes(21個)、l_hand(1個)、r_hand(1個)、velocity(1個)、angular(1個)
# 各フレームごとにまとめる
for frame_idx in range(num_images):
    combined_data = []

    # joint_changeを追加
    for joint_idx in range(num_joints):
        joint_change_all_frames = np.load(os.path.join(output_action_data_dir, f"joint_{joint_idx:02d}.npy"))
        combined_data.append(joint_change_all_frames[frame_idx])

    # 手の状態を追加
    l_hand = np.load(os.path.join(output_action_data_dir, "l_hand_closure.npy"))[frame_idx]
    r_hand = np.load(os.path.join(output_action_data_dir, "r_hand_closure.npy"))[frame_idx]
    combined_data.extend([l_hand, r_hand])

    # 速度と角速度を追加
    velocity = velocity_data[frame_idx]
    angular_velocity = angular_velocity_data[frame_idx]
    combined_data.extend([velocity, angular_velocity])

    # 保存
    combined_output_path = os.path.join(action_data_combined_dir, f"frame_{frame_idx:06d}.npy")
    np.save(combined_output_path, np.array(combined_data))

    print(f"Frame {frame_idx} action data saved to {combined_output_path}")

#正規化バージョンのデータセット作成
#joint_changes_normalized(21個)、l_hand(1個)、r_hand(1個)、velocity_normalized(1個)、angular_velocity_normalized(1個)
for frame_idx in range(num_images):
    normalized_data = []

    # joint_changeを追加
    for joint_idx in range(num_joints):
        joint_change_all_frames = np.load(os.path.join(output_action_data_dir, f"joint_{joint_idx:02d}.npy"))
        normalized_data.append(joint_change_all_frames[frame_idx])

    # 手の状態を追加（元から0→1)
    l_hand = np.load(os.path.join(output_action_data_dir, "l_hand_closure.npy"))[frame_idx]
    r_hand = np.load(os.path.join(output_action_data_dir, "r_hand_closure.npy"))[frame_idx]
    normalized_data.extend([l_hand, r_hand])

    # 速度と加速度の正規化データを追加
    velocity_normalized = velocity_normalized[frame_idx]    
    angular_velocity_normalized = angular_velocity_normalized[frame_idx]
    normalized_data.extend([velocity_normalized, angular_velocity_normalized])

    # Save normalized data
    normalized_data_path = os.path.join(combined_data_normalized_dir, f"frame_{frame_idx:06d}_normalized.npy")
    np.save(normalized_data_path, normalized_data)

    print(f"Frame {frame_idx} action data saved to {normalized_data_path}")
