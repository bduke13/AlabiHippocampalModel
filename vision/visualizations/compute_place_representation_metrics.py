import numpy as np
import csv
import os

# 从 vis_utils 模块中导入加载数据的函数和输出目录变量
from vis_utils import load_hmaps, convert_xzy_hmaps, OUTPUT_DIR

def compute_cosine_similarity_metrics(hmap_pcn, hmap_loc):
    """
    Compute cosine similarity between consecutive place cell activation vectors and
    the corresponding Euclidean distance between positions.

    Args:
        hmap_pcn: numpy array of shape (num_steps, num_place_cells) representing place cell activations.
        hmap_loc: numpy array of shape (num_steps, 3) representing the robot positions (X, Y, Z).

    Returns:
        distances: numpy array of Euclidean distances between adjacent positions.
        cosine_similarities: numpy array of cosine similarities between adjacent activation vectors.
    """
    num_steps = hmap_pcn.shape[0]
    distances = []
    cosine_similarities = []

    for i in range(num_steps - 1):
        # 取出相邻时间步的激活向量
        vec1 = hmap_pcn[i]
        vec2 = hmap_pcn[i + 1]
        # 计算余弦相似度（加一个小 epsilon 防止除零）
        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
        cosine_similarities.append(cos_sim)

        # 使用 X 和 Z 坐标计算欧氏距离（假设 hmap_loc 中 index 0 为 X，index 2 为 Z）
        pos1 = np.array([hmap_loc[i, 0], hmap_loc[i, 2]])
        pos2 = np.array([hmap_loc[i + 1, 0], hmap_loc[i + 1, 2]])
        dist = np.linalg.norm(pos1 - pos2)
        distances.append(dist)

    return np.array(distances), np.array(cosine_similarities)


def save_metrics_to_csv(distances, cosine_similarities, output_path):
    """
    Save detailed distances and cosine similarity data to a CSV file.

    Args:
        distances: numpy array of distances.
        cosine_similarities: numpy array of cosine similarities.
        output_path: Path of the CSV file to save.
    """
    with open(output_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Distance', 'Cosine Similarity'])
        for d, cs in zip(distances, cosine_similarities):
            writer.writerow([d, cs])
    print(f"Detailed metrics saved to {output_path}")


def save_summary_metrics_to_csv(avg_cos_sim, std_cos_sim, avg_distance, corr_coef, output_path):
    """
    Save summary metrics (average cosine similarity, standard deviation, average distance,
    and correlation coefficient) to a CSV file.

    Args:
        avg_cos_sim: Average cosine similarity.
        std_cos_sim: Standard deviation of cosine similarities.
        avg_distance: Average Euclidean distance.
        corr_coef: Correlation coefficient between distance and cosine similarity.
        output_path: Path of the CSV file to save.
    """
    with open(output_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ['Average Cosine Similarity', 'Cosine Similarity Std', 'Average Distance', 'Correlation Coefficient'])
        writer.writerow([avg_cos_sim, std_cos_sim, avg_distance, corr_coef])
    print(f"Summary metrics saved to {output_path}")


def main():
    # Load hmap data (assumes vis_utils.load_hmaps returns two arrays: hmap_loc and hmap_pcn)
    hmap_loc, hmap_pcn = load_hmaps(hmap_names=["hmap_loc", "hmap_pcn"])
    # If needed, decompose hmap_loc into individual coordinate arrays
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)

    # 计算每个时间步相邻位置之间的欧氏距离和余弦相似度
    distances, cosine_similarities = compute_cosine_similarity_metrics(hmap_pcn, hmap_loc)

    # 保存详细数据到 CSV 文件
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    detailed_csv_path = os.path.join(OUTPUT_DIR, "place_representation_metrics.csv")
    save_metrics_to_csv(distances, cosine_similarities, detailed_csv_path)

    # 计算 summary 指标
    avg_cos_sim = np.mean(cosine_similarities)
    std_cos_sim = np.std(cosine_similarities)
    avg_distance = np.mean(distances)
    if len(distances) > 1:
        corr_coef = np.corrcoef(distances, cosine_similarities)[0, 1]
    else:
        corr_coef = np.nan

    # 保存 summary 指标到另一 CSV 文件
    summary_csv_path = os.path.join(OUTPUT_DIR, "place_representation_summary_metrics.csv")
    save_summary_metrics_to_csv(avg_cos_sim, std_cos_sim, avg_distance, corr_coef, summary_csv_path)


if __name__ == "__main__":
    main()