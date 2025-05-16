import csv
import re
import os
import random
import logging
import numpy as np
from sklearn.model_selection import train_test_split
import json
import pandas as pd
from ase.io import read
from dscribe.descriptors import SOAP
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SymmetryUndetermined
from pymatgen.core import Structure
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

#properties = ["dielectric", "elasticity", "perovskites", "jdft2d", "supercon3d", "mp_gap"]
properties = ["dielectric"]
n_clusters = [5, 10, 20, 30, 40, 50]

def setup_logging(property, n_cluster):
    # 创建日志目录
    log_dir = f'../logs/{property}/{n_cluster}'
    os.makedirs(log_dir, exist_ok=True)

    # 配置日志
    log_file = f'{log_dir}/{property}_LOCO{n_cluster}Clusters.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # 保存到文件
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )

def safe_formula_extraction(cif_path):
    with open(cif_path, 'r') as f:
        content = f.read()
        # 优化后的正则表达式
        pattern = r"_chemical_formula_sum\s+(?:(['\"])(.*?)\1|(\S+))(?=\s|$)"
        match = re.search(pattern, content)

        if match:
            # 判断是否为引号模式
            if match.group(1):  # 引号模式
                raw = match.group(2)
                formula = raw.replace(" ", "")
            else:  # 无引号模式
                formula = match.group(3)
            # 清理非法字符
            return re.sub(r"[^\w()\-]", "", formula)
    return None

def main():
    seed = 42
    for n_cluster in n_clusters:
        for property in properties:
            setup_logging(property, n_cluster)
            logging.info(f"Processing dataset: {property}")

            random.seed(seed)
            np.random.seed(seed)

            target_file = f'../data/{property}/targets.csv'
            if not os.path.exists(target_file):
                logging.error(f"{target_file} not found, skipping {property}")
                continue

            indexs=[]
            values=[]
            with open(target_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    index = int(row[0])
                    value = float(row[1])
                    indexs.append(index)
                    values.append(value)

            all_species = set()
            n_species = []
            struc_dict = {
                "structures": [],
                "systems": [],
                "valid_cif_ids": []
            }
            for cif_id in range(len(indexs)):
                cif_path = os.path.join(f'../data/{property}', f'{cif_id}.cif')
                if not os.path.exists(cif_path):
                    logging.warning(f"{cif_path} not found, skipping this sample")
                    continue
                try:
                    structure = Structure.from_file(cif_path)
                    system = read(cif_path)
                    struc_dict["structures"].append(structure)
                    struc_dict["systems"].append(system)
                    species = {site.specie.symbol for site in structure}
                    n_species.append(len(species))
                    all_species.update(species)
                    struc_dict["valid_cif_ids"].append(cif_id)
                except RuntimeError as e:
                    logging.warning(f"Skipping cif_id {cif_id} due to RuntimeError: {str(e)}")
                    continue
            n_max = max(n_species)
            all_species = sorted(list(all_species))  # 转换为有序列表
            logging.info(f"{property} global species: {len(all_species)}, single max length: {n_max}")

            all_features = []
            for structure, system in zip(struc_dict["structures"], struc_dict["systems"]):
                species = {site.specie.symbol for site in structure}
                single_species = set(species)
                single_species = sorted(list(single_species))
                diff = set(all_species) - set(single_species)
                ex_species = sorted(list(diff))
                selected_diff = random.sample(ex_species, n_max-len(single_species))
                combined_species = single_species + selected_diff
                combined_species = sorted(combined_species)
                elements = system.get_chemical_symbols()

                soap = SOAP(
                    species=combined_species,
                    r_cut=8.0,
                    n_max=6,
                    l_max=4,
                    periodic=True,
                    # average="inner"
                )
                dim = soap.get_number_of_features()
                feature_vector = np.zeros((len(all_species), dim))  # 形状: (M, D)，例如 (20, 252)
                soap_features = soap.create(system)

                for elem_idx, elem in enumerate(all_species):
                    elem_mask = np.array(elements) == elem
                    if np.any(elem_mask):
                        elem_soap = soap_features[elem_mask]
                        feature_vector[elem_idx] = np.mean(elem_soap, axis=0)
                all_features.append(feature_vector.flatten())
            logging.info(f"Processed length for {property}: {len(all_features)}")
            # 转换为 NumPy 数组
            all_features = np.array(all_features)  # 形状: (n_samples, n_features)
            logging.info(f"all_features shape for {property}: {all_features.shape}")  # 形状: (4764, 658560)
            #all_features shape:
            # pca = PCA(n_components=1024)
            # all_features = pca.fit_transform(all_features)
            kmeans = KMeans(n_clusters=n_cluster, max_iter=500, random_state=222)
            kmeans.fit(all_features)
            labels = kmeans.labels_
            max_index = len(all_features)
            fold_labels = np.full(max_index, -1, dtype=int)
            cluster = {}
            for i, x in enumerate(labels):
                fold_labels[i] = int(x)
                if x in cluster:
                    cluster[int(x)].append(struc_dict["valid_cif_ids"][i])
                else:
                    cluster[int(x)] = [struc_dict["valid_cif_ids"][i]]
            for key in sorted(cluster):
                logging.info(f"Cluster {key}: {len(cluster[key])} samples")

            os.makedirs(f'../folds/{property}_folds/test', exist_ok=True)
            with open(f'../folds/{property}_folds/test/SOAP_{property}_LOCO_target_clusters{n_cluster}_test.json', 'w') as file:
                json.dump(cluster, file)

            tsne = TSNE(n_components=2, random_state=42)
            tsne_features = tsne.fit_transform(all_features)
            X_tsne = pd.DataFrame(tsne_features, columns=["X", "Y"]).reset_index(drop=True)

            with open(f"../folds/{property}_folds/test/SOAP_{property}_LOCO_target_clusters{n_cluster}_test.json") as f:
                folds_dict = json.load(f)

            def create_custom_colormap(n_folds):
                """创建适用于多类别的彩虹色系颜色映射"""
                base_cmap = colormaps.get_cmap('tab20')  # 'tab20'
                colors = base_cmap(np.linspace(0, 1, n_folds))
                return ListedColormap(colors)
            cmap = create_custom_colormap(n_cluster)

            # Plot the reduced-dimensional data
            plt.rcParams.update({
                'axes.labelweight': 'bold',  # 坐标轴标签加粗[4](@ref)
                'axes.titleweight': 'bold',  # 标题加粗[4](@ref)
                'xtick.labelsize': 20,  # X轴刻度字号
                'ytick.labelsize': 20,  # Y轴刻度字号
                'axes.labelsize': 22,  # 坐标轴标签字号[2](@ref)
                'axes.titlesize': 24  # 标题字号[2](@ref)
            })

            plt.figure(figsize=(12, 12))
            plt.scatter(X_tsne.iloc[:, 0], X_tsne.iloc[:, 1], c=fold_labels, cmap=cmap, s=600, linewidths=6, marker='+', )
            plt.xlabel('t-SNE 1',fontsize=24)
            plt.ylabel('t-SNE 2',fontsize=24)
            #plt.tick_params(axis='both', which='major', width=2)  # 刻度线加粗[4](@ref)
            # plt.title('t-SNE Dimensionality Reduction')
            # plt.show()
            plt.tight_layout()
            plt.savefig(f'{property}_SOAP_LOCO{n_cluster}.png', dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Visualization completed for {property}, saved as {property}_SOAP_LOCO{n_cluster}.png")
            # batch_df = pd.DataFrame({
            #     "0": list(range(len(all_properties))),
            #     **{str(i + 1): [f[i] for f in all_features] for i in range(len(all_features[0]))},
            #     "property": all_properties
            # })
            #
            # # 保存当前批次
            # batch_df.to_csv(f"SOAP_MiddleFeature_dielectric_property.csv", index=False)
            # print(f"Processed {len(all_properties)} materials")

            # 训练/验证集生成
            all_indices = set(struc_dict["valid_cif_ids"])
            train_dict = {}
            val_dict = {}
            # 遍历每一折（每个标签作为测试集）
            for label, test_indices in folds_dict.items():
                # 剩余样本索引
                remaining = list(all_indices - set(test_indices))

                # 随机分割（训练:验证=4:1）
                val_size = max(1, int(len(remaining) * 0.2))  # 验证集至少1个
                train_indices, val_indices = train_test_split(
                    remaining, test_size=val_size, random_state=42
                )
                train_dict[label] = train_indices
                val_dict[label] = val_indices

            train_pair = next(iter(train_dict.items()))  # 直接获取第一个键值对元组
            logging.info(f"First train key for {property}: {train_pair[0]}, length: {len(train_pair[1])}")
            val_pair = next(iter(val_dict.items()))  # 直接获取第一个键值对元组
            logging.info(f"First val key for {property}: {val_pair[0]}, length: {len(val_pair[1])}")

            def sort_dict_by_key(dct):
                # 按字符串键的数值升序排列（例如 "35" -> 35）
                sorted_items = sorted(dct.items(), key=lambda x: int(x[0]))
                return {k: v for k, v in sorted_items}

            def save_sorted_dict(sorted_dict, output_path):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(sorted_dict, f)

            # 按数值排序键
            sorted_train = sort_dict_by_key(train_dict)
            sorted_val = sort_dict_by_key(val_dict)

            # 保存为JSON文件
            save_sorted_dict(sorted_train, f"../folds/{property}_folds/train/SOAP_{property}_LOCO_target_clusters{n_cluster}_train.json")
            save_sorted_dict(sorted_val, f"../folds/{property}_folds/val/SOAP_{property}_LOCO_target_clusters{n_cluster}_val.json")
            logging.info(f"Train/val splits saved for {property}")


if __name__ == "__main__":
    main()
