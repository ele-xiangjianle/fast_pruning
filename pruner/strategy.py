import torch
from typing import Sequence
from tqdm import tqdm
import random
import numpy as np
import heapq
import math
import gurobipy as grb
import time
import json
import pyomo.environ as pyo


class FDMStrategy():
    def __init__(self) -> None:
        pass

    @staticmethod
    def constr_dist_matrix(n, w):
        matrix = torch.zeros([n, n])
        for i in range(n):
            for j in range(n):
                matrix[i, j] = torch.linalg.norm(w[i] - w[j], ord=1).item()
        return matrix

    @staticmethod
    def constr_best_pair(n, matrix):
        max_dist = 0
        best_pair = ()
        for i in range(n):
            for j in range(i+1, n):
                if matrix[i, j] > max_dist:
                    max_dist = matrix[i, j].item()
                    best_pair = (i, j)
        return best_pair

    @staticmethod
    def fdm(n, m, w):
        print(f"{n} --> {m}")
        matrix = FDMStrategy.constr_dist_matrix(n, w)
        best_pair = FDMStrategy.constr_best_pair(n, matrix)
        save_points = set()
        save_points.add(best_pair[0])
        save_points.add(best_pair[1])

        while len(save_points) < m:
            max_min_dist = 0
            max_min_point = None
            for i in range(n):
                if i in save_points:
                    continue
                cur_min_dist = float('inf')
                for j in save_points:
                    if matrix[i, j].item() < cur_min_dist:
                        cur_min_dist = matrix[i, j].item()
                if cur_min_dist > max_min_dist:
                    max_min_dist = cur_min_dist
                    max_min_point = i
            save_points.add(max_min_point)

        universe = set([i for i in range(n)])
        pruned_points = universe - save_points
        return pruned_points

class kSC():
    def __init__(self) -> None:
        pass

    @staticmethod
    def constr_dist_matrix(n, w):
        matrix = torch.zeros([n, n])
        for i in range(n):
            for j in range(n):
                matrix[i, j] = torch.linalg.norm(w[i] - w[j], ord=1).item()
        return matrix

    @staticmethod
    def constr_closest_pair(n, matrix):
        min_dist = float('inf')
        closest_pair = ()
        for i in range(n):
            for j in range(i+1, n):
                if matrix[i, j] < min_dist:
                    min_dist = matrix[i, j].item()
                    closest_pair = (i, j)
        return closest_pair
    @staticmethod
    def kSC(n, m, w):
        print(f"{n} --> {m}")
        matrix = kSC.constr_dist_matrix(n, w)
        closest_pair = kSC.constr_closest_pair(n, matrix)
        save_points = set()
        save_points.add(closest_pair[0])
        save_points.add(closest_pair[1])

        while len(save_points) < m:
            min_max_dist = float('inf')
            min_max_point = None
            for i in range(n):
                if i in save_points:
                    continue
                cur_max_dist = 0
                for j in save_points:
                    if matrix[i, j].item() > cur_max_dist:
                        cur_max_dist = matrix[i, j].item()
                if cur_max_dist < min_max_dist:
                    min_max_dist = cur_max_dist
                    min_max_point = i
            save_points.add(min_max_point)

        universe = set([i for i in range(n)])
        pruned_points = universe - save_points
        return pruned_points

class ECPFDMStrategy():
    def __init__(self):
        pass
    def apply(self, weights, ratio):
        if ratio >= 1:
            return []
        n = len(weights)
        w = weights.view(n, -1)
        n_to_prune = int(ratio* n)
        if n_to_prune == 0:
            return []
        
        indices = FDMStrategy.fdm(n, n - n_to_prune, w)
        
        return indices



class PCCFDMStrategy():
    def __init__(self, convs, ratio):
        self.weights = []
        self.total = 0

        for conv in convs:
            n = len(conv.weight)  # out_channels
            self.weights.append(conv.weight.view(n, -1))
            self.total += n

        self.n_to_prune = int(ratio * self.total)

        self.save_indices = []
        self.dist_matrices = []
        self.max_dists = []
        self.min_dists = []

        print("Init dist matrices")
        self.init_dist_matrices()
        self.init_indices()

        # 保持你原来的行为：每层先走一步（扩展一个点）
        for idx in range(len(self.dist_matrices)):
            self.step(idx)

    @staticmethod
    def max_dist_pair(matrix):
        max_dist = 0.0
        best_pair = ()
        n = len(matrix)
        for i in range(n):
            for j in range(i + 1, n):
                d = matrix[i, j].item()
                if d > max_dist:
                    max_dist = d
                    best_pair = (i, j)
        return best_pair, max_dist

    def init_dist_matrices(self):
        bar_size = 0
        for w in self.weights:
            bar_size += len(w) ** 2

        pbar = tqdm(total=bar_size)
        for w in self.weights:
            n = len(w)
            matrix = torch.zeros([n, n])
            for i in range(n):
                for j in range(n):
                    matrix[i, j] = torch.linalg.norm(w[i] - w[j], ord=2).item()
                    pbar.update(1)
            self.dist_matrices.append(matrix)
        pbar.close()

    def init_indices(self):
        for idx in range(len(self.weights)):
            matrix = self.dist_matrices[idx]
            pair, max_dist = PCCFDMStrategy.max_dist_pair(matrix)

            # 如果某层 n=1，pair 可能是空
            if not pair:
                self.save_indices.append(set([0]))
                self.max_dists.append(0.0)
                self.min_dists.append(0.0)
            else:
                self.save_indices.append(set(pair))
                self.max_dists.append(max_dist)
                self.min_dists.append(max_dist)

    def pruned_cnt(self):
        total_save = 0
        for s in self.save_indices:
            total_save += len(s)
        return self.total - total_save

    def _next_step(self, idx):
        """
        预演一步 step（不修改状态），返回 (next_min_dist, next_point)
        """
        indices = self.save_indices[idx]
        matrix = self.dist_matrices[idx]
        n = len(matrix)

        if len(indices) >= n:
            return self.min_dists[idx], None

        max_min_dist = 0.0
        max_min_point = None

        for i in range(n):
            if i in indices:
                continue
            cur_min_dist = float("inf")
            for j in indices:
                d = matrix[i, j].item()
                if d < cur_min_dist:
                    cur_min_dist = d
            if cur_min_dist > max_min_dist:
                max_min_dist = cur_min_dist
                max_min_point = i

        return max_min_dist, max_min_point

    def step(self, idx):
        """
        真正走一步（会调用一次 _next_step）
        """
        next_min_dist, next_point = self._next_step(idx)
        if next_point is None:
            return
        self.save_indices[idx].add(next_point)
        self.min_dists[idx] = next_min_dist

    def step_with_cache(self, idx, next_min_dist, next_point):
        """
        用“缓存的前瞻结果”直接落地（避免重复 _next_step）
        """
        if next_point is None:
            return
        self.save_indices[idx].add(next_point)
        self.min_dists[idx] = next_min_dist

    def _entropy_from_min(self, idx, min_dist):
        denom = self.max_dists[idx]
        return 0.0 if denom == 0 else (min_dist / denom)

    def run(self):
        print("Run")
        L = len(self.weights)

        # 目标循环次数（可能为 0）
        total_iters = max(0, self.pruned_cnt() - self.n_to_prune)
        pbar = tqdm(total=total_iters)

        # ====== 缓存每层“再 step 一次”的前瞻结果 ======
        self._ver = [0] * L  # 堆的懒更新版本号
        next_point = [None] * L
        next_min_dist = [0.0] * L
        next_entropy = [0.0] * L

        def refresh(idx):
            nm, np = self._next_step(idx)
            next_min_dist[idx] = nm
            next_point[idx] = np
            next_entropy[idx] = self._entropy_from_min(idx, nm)

        # 最大堆：(-next_entropy, idx, ver)
        heap = []
        for idx in range(L):
            if len(self.save_indices[idx]) >= len(self.dist_matrices[idx]):
                continue
            refresh(idx)
            heapq.heappush(heap, (-next_entropy[idx], idx, self._ver[idx]))

        # ====== 主循环：每轮只更新“被选中的层”的前瞻 ======
        while self.pruned_cnt() != self.n_to_prune:
            # 取有效最大项（丢弃过期项）
            while heap:
                neg_e, idx, ver = heap[0]
                if ver == self._ver[idx]:
                    break
                heapq.heappop(heap)

            if not heap:
                # 所有层都“满了”但仍没达到目标，说明 ratio 设得太大（不可行）
                print("[Warn] No selectable layer left. Target prune count may be infeasible.")
                break

            _, pick_idx, _ = heapq.heappop(heap)

            # 用缓存结果落地（这一轮不再重复 _next_step）
            self.step_with_cache(pick_idx, next_min_dist[pick_idx], next_point[pick_idx])
            pbar.update(1)

            # 只刷新这一层并重新入堆
            if len(self.save_indices[pick_idx]) < len(self.dist_matrices[pick_idx]):
                self._ver[pick_idx] += 1
                refresh(pick_idx)
                heapq.heappush(heap, (-next_entropy[pick_idx], pick_idx, self._ver[pick_idx]))

        pbar.close()

        # ====== 输出 prune_indices（与原逻辑一致）======
        prune_indices = []
        info_entropy_array = []

        for idx in range(L):
            universe = set(range(len(self.weights[idx])))
            prune_indices.append(universe - self.save_indices[idx])

            info_entropy = self._entropy_from_min(idx, self.min_dists[idx])
            info_entropy_array.append(info_entropy)

            layer_cnt = len(universe)
            pruned_cnt = len(prune_indices[idx])
            print(
                f"Conv: {idx} \t  {layer_cnt} --> {layer_cnt - pruned_cnt}\t "
                f"Ratio: {'%.3f' % (100 * pruned_cnt / layer_cnt)}%\t\tLayer Diversity {info_entropy} "
            )

        if len(info_entropy_array) > 0:
            t = torch.tensor(info_entropy_array)
            print(f"\nNetwork Diversity: {t.min().item()}")

        print(f"Total: {self.total}\tNeed to Prune: {self.n_to_prune}\n")
        return prune_indices


class C_SEDP():
    """
    C-SEDP 快速实现：预计算 e_{ij}=exp(-beta*d_{ij})，维护 H[x]=sum_{y in S} e_{x,y}

    层内选择 O(N)：每步选 H[x] 最小的未选点
    跨层选择 O(log L)：最小堆按 Delta_l 排序，仅更新被修改层

    Delta_l = exp(-gamma * n_l^new) - exp(-gamma * n_l^old)
    D_net 提升正比于 Delta_l 的绝对值，且 Delta_l 仅依赖本层状态
    """

    def __init__(self, convs, ratio, beta=1.0, gamma=1.0):
        self.weights = []
        self.total = 0
        for conv in convs:
            n = len(conv.weight)
            self.weights.append(conv.weight.view(n, -1))
            self.total += n

        self.n_to_prune = int(ratio * self.total)
        self.beta = float(beta) if beta > 0 else 1.0
        self.gamma = float(gamma) if gamma > 0 else 1.0
        self.L = len(self.weights)

        self.e_matrices = []       # e_{ij} = exp(-beta * d_{ij})，对角线=0
        self.H = []                # H[x] = sum_{y in S} e_{x,y}
        self.selected = []         # set of saved indices per layer
        self.Z = []                # Z = sum_{i<j, i,j in S} e_{ij}
        self.D_full = []           # D_l(F_l) full-set soft entropy
        self.n_l = []              # normalized D_l / D_full
        self.best_cand = []        # index of best unselected candidate
        self.best_H = []           # H value of best candidate

        print("Init distance & e-matrices for C-SEDP")
        self._init_matrices()
        self._init_selected()
        self._init_H_and_heap()

    # ==================== 初始化 ====================

    def _init_matrices(self):
        """距离矩阵 + e 矩阵"""
        bar_size = sum(len(w) ** 2 for w in self.weights)
        pbar = tqdm(total=bar_size, desc="e-matrix")
        for w in self.weights:
            n = len(w)
            dist = torch.zeros([n, n])
            for i in range(n):
                for j in range(n):
                    dist[i, j] = torch.linalg.norm(w[i] - w[j], ord=2).item()
                    pbar.update(1)
            e_mat = torch.exp(-self.beta * dist)
            e_mat.fill_diagonal_(0.0)  # i<j 的 pair 不含自环
            self.e_matrices.append(e_mat)
        pbar.close()

    def _init_selected(self):
        """每层初始：选距离最远的两个 filter（e_{ij} 最小）"""
        for idx in range(self.L):
            e_mat = self.e_matrices[idx]
            n = len(e_mat)

            if n <= 1:
                self.selected.append(set(range(n)))
                self.D_full.append(0.0)
                self.Z.append(0.0)
                self.n_l.append(0.0)
                continue

            min_e = float('inf')
            best_pair = ()
            for i in range(n):
                for j in range(i + 1, n):
                    if e_mat[i, j].item() < min_e:
                        min_e = e_mat[i, j].item()
                        best_pair = (i, j)

            sel = set(best_pair)
            self.selected.append(sel)

            Z = e_mat[best_pair[0], best_pair[1]].item()
            self.Z.append(Z)

            Z_full = torch.triu(e_mat, diagonal=1).sum().item()
            D_full = (-1.0 / self.beta) * math.log(Z_full) if Z_full > 0 else 0.0
            self.D_full.append(D_full)

            D_l = (-1.0 / self.beta) * math.log(Z) if Z > 0 else 0.0
            self.n_l.append(D_l / D_full if D_full > 0 else 0.0)

    def _init_H_and_heap(self):
        """初始化 H 数组和跨层最大堆"""
        self.heap = []  # (delta_l, idx)

        for idx in range(self.L):
            e_mat = self.e_matrices[idx]
            n = len(e_mat)
            sel = self.selected[idx]

            if n <= 1 or len(sel) >= n:
                self.H.append(torch.zeros(n))
                self.best_cand.append(None)
                self.best_H.append(0.0)
                continue

            H = torch.zeros(n)
            sel_list = list(sel)
            for y in range(n):
                if y not in sel:
                    H[y] = sum(e_mat[i, y].item() for i in sel_list)
            self.H.append(H)

            best_val, best_c = self._find_best(idx)
            self.best_H.append(best_val)
            self.best_cand.append(best_c)

            if best_c is not None:
                delta = self._compute_delta(idx, best_val)
                heapq.heappush(self.heap, (delta, idx))

    # ==================== 辅助 ====================

    def _find_best(self, idx):
        """O(N) 找层内 H 最小的未选点"""
        sel = self.selected[idx]
        H_arr = self.H[idx]
        n = len(H_arr)
        best_val = float('inf')
        best_idx = None
        for y in range(n):
            if y not in sel:
                val = H_arr[y].item()
                if val < best_val:
                    best_val = val
                    best_idx = y
        return best_val, best_idx

    def _compute_delta(self, idx, cand_H):
        """
        Delta_l = exp(-gamma * n_l^new) - exp(-gamma * n_l^old)
        表示加入候选点后，多样性提升的收益。
        """
        Z_old = self.Z[idx]
        Z_new = Z_old + cand_H
        if Z_new <= 0:
            return 0.0

        n_old = self.n_l[idx]
        D_full = self.D_full[idx]
        if D_full > 0:
            n_new = (-1.0 / self.beta) * math.log(Z_new) / D_full
        else:
            n_new = 0.0

        return math.exp(-self.gamma * n_new) - math.exp(-self.gamma * n_old)

    # ==================== 主循环 ====================

    def run(self):
        total_to_keep = self.total - self.n_to_prune
        current_kept = sum(len(s) for s in self.selected)
        steps_needed = max(0, total_to_keep - current_kept)

        if steps_needed == 0:
            return self._build_output()

        pbar = tqdm(total=steps_needed, desc="C-SEDP")

        for _ in range(steps_needed):
            # 弹出最佳层（lazy deletion）
            idx = None
            while self.heap:
                delta, heap_idx = heapq.heappop(self.heap)
                if (self.best_cand[heap_idx] is not None
                        and self.best_cand[heap_idx] not in self.selected[heap_idx]):
                    idx = heap_idx
                    break

            if idx is None:
                break

            best_c = self.best_cand[idx]
            e_mat = self.e_matrices[idx]
            n = len(e_mat)
            sel = self.selected[idx]

            # 添加候选点
            sel.add(best_c)
            self.Z[idx] += self.best_H[idx]

            # 更新 H 数组 O(N)
            H_arr = self.H[idx]
            for y in range(n):
                if y not in sel:
                    H_arr[y] += e_mat[best_c, y].item()

            # 更新 n_l
            Z_new = self.Z[idx]
            D_full = self.D_full[idx]
            if D_full > 0 and Z_new > 0:
                D_new = (-1.0 / self.beta) * math.log(Z_new)
                self.n_l[idx] = D_new / D_full

            # 找新最佳候选
            best_val, new_best_c = self._find_best(idx)
            self.best_H[idx] = best_val
            self.best_cand[idx] = new_best_c

            # 计算新 Delta_l 入堆
            if new_best_c is not None:
                delta = self._compute_delta(idx, best_val)
                heapq.heappush(self.heap, (delta, idx))

            pbar.update(1)

        pbar.close()
        return self._build_output()

    # ==================== 输出 ====================

    def _build_output(self):
        prune_indices = []
        info_entropy_array = []

        for idx in range(self.L):
            universe = set(range(len(self.weights[idx])))
            prune_indices.append(universe - self.selected[idx])
            info_entropy_array.append(self.n_l[idx])

            layer_cnt = len(universe)
            pruned_cnt = len(prune_indices[idx])
            print(
                f"Conv: {idx} \t  {layer_cnt} --> {layer_cnt - pruned_cnt}\t "
                f"Ratio: {100 * pruned_cnt / max(1, layer_cnt):.3f}%\t\t"
                f"Layer Diversity {self.n_l[idx]:.6f}"
            )

        if len(info_entropy_array) > 0:
            t = torch.tensor(info_entropy_array, dtype=torch.float32)
            print(f"\nNetwork Diversity (mean normalized D_l): {t.mean().item():.6f}")

            energy = torch.tensor(self.n_l, dtype=torch.float32)
            D_net = (-1.0 / self.gamma) * torch.logsumexp(
                -self.gamma * energy, dim=0).item()
            print(f"Network Diversity D_net: {D_net:.6f}")

        print(f"Total: {self.total}\tNeed to Prune: {self.n_to_prune}")
        print(f"Total kept: {self.total - self.n_to_prune}\n")
        return prune_indices

class PCCFDMGurobiStrategy():
    def __init__(self, convs, ratio):
        self.weights = []
        self.total = 0

        for conv in convs:
            n = len(conv.weight)
            self.weights.append(conv.weight.view(n, -1))
            self.total += n

        self.n_to_prune = int(ratio * self.total)

        self.save_indices = []
        self.dist_matrices = []
        self.max_dists = []
        self.min_dists = []

        print("Init dist matrices")
        self.init_dist_matrices()
        self.init_indices()

        for idx in range(len(self.dist_matrices)):
            self.step_gurobi(idx)

    @staticmethod
    def max_dist_pair(matrix):
        max_dist = 0.0
        best_pair = ()
        n = len(matrix)
        for i in range(n):
            for j in range(i + 1, n):
                d = matrix[i, j].item()
                if d > max_dist:
                    max_dist = d
                    best_pair = (i, j)
        return best_pair, max_dist

    def init_dist_matrices(self):
        bar_size = 0
        for w in self.weights:
            bar_size += len(w) ** 2

        pbar = tqdm(total=bar_size)
        for w in self.weights:
            n = len(w)
            matrix = torch.zeros([n, n])
            for i in range(n):
                for j in range(n):
                    matrix[i, j] = torch.linalg.norm(w[i] - w[j], ord=2).item()
                    pbar.update(1)
            self.dist_matrices.append(matrix)
        pbar.close()

    def init_indices(self):
        for idx in range(len(self.weights)):
            matrix = self.dist_matrices[idx]
            pair, max_dist = PCCFDMGurobiStrategy.max_dist_pair(matrix)

            if not pair:
                self.save_indices.append(set([0]))
                self.max_dists.append(0.0)
                self.min_dists.append(0.0)
            else:
                self.save_indices.append(set(pair))
                self.max_dists.append(max_dist)
                self.min_dists.append(max_dist)

    def pruned_cnt(self):
        total_save = 0
        for s in self.save_indices:
            total_save += len(s)
        return self.total - total_save

    def step_gurobi(self, idx):
        max_min_dist = float("inf")
        max_min_point = None
        indices = self.save_indices[idx]
        matrix = self.dist_matrices[idx]
        n = len(matrix)
        print('filter index:', idx, 'n=:', n, 'm=:', len(indices) + 1)
        if len(indices) == n:
            return

        grb_model = grb.Model(name="MIP Model")
        indices_x = [grb_model.addVar(vtype=grb.GRB.BINARY) for i in range(n)]
        d = grb_model.addVar(vtype=grb.GRB.CONTINUOUS, name='diversity')

        objective = d

        for i in range(n):
            for j in range(n):
                if i != j:
                    grb_model.addConstr(d <= 10 * (2 - indices_x[i] - indices_x[j]) + matrix[i, j])

        grb_model.addConstr(
            lhs=grb.quicksum(indices_x),
            sense=grb.GRB.EQUAL,
            rhs=len(indices) + 1
        )

        grb_model.ModelSense = grb.GRB.MAXIMIZE
        grb_model.setObjective(objective)
        grb_model.Params.LogToConsole = 0

        grb_model.optimize()

        save_point = []
        for i in range(n):
            if indices_x[i].x >= 0.99:
                save_point.append(i)

        max_min_dist = d.x

        self.save_indices[idx] = set(save_point)
        self.min_dists[idx] = max_min_dist

    def _entropy_from_min(self, idx, min_dist):
        denom = self.max_dists[idx]
        return 0.0 if denom == 0 else (min_dist / denom)

    def run(self):
        print("Run")
        L = len(self.weights)

        total_iters = max(0, self.pruned_cnt() - self.n_to_prune)
        pbar = tqdm(total=total_iters)

        while self.pruned_cnt() != self.n_to_prune:
            max_entropy = -1.0
            max_idx = None

            for idx in range(L):
                if len(self.save_indices[idx]) >= len(self.dist_matrices[idx]):
                    continue

                indices = self.save_indices[idx]
                matrix = self.dist_matrices[idx]
                n = len(matrix)

                grb_model = grb.Model(name="MIP Model")
                indices_x = [grb_model.addVar(vtype=grb.GRB.BINARY) for i in range(n)]
                d = grb_model.addVar(vtype=grb.GRB.CONTINUOUS, name='diversity')

                objective = d

                for i in range(n):
                    for j in range(n):
                        if i != j:
                            grb_model.addConstr(d <= 10 * (2 - indices_x[i] - indices_x[j]) + matrix[i, j])

                grb_model.addConstr(
                    lhs=grb.quicksum(indices_x),
                    sense=grb.GRB.EQUAL,
                    rhs=len(indices) + 1
                )

                grb_model.ModelSense = grb.GRB.MAXIMIZE
                grb_model.setObjective(objective)
                grb_model.Params.LogToConsole = 0
                grb_model.optimize()

                next_min_dist = d.x
                next_entropy = self._entropy_from_min(idx, next_min_dist)

                if next_entropy > max_entropy:
                    max_entropy = next_entropy
                    max_idx = idx

            if max_idx is None:
                print("[Warn] No selectable layer left. Target prune count may be infeasible.")
                break

            self.step_gurobi(max_idx)
            pbar.update(1)

        pbar.close()

        prune_indices = []
        info_entropy_array = []

        for idx in range(L):
            universe = set(range(len(self.weights[idx])))
            prune_indices.append(universe - self.save_indices[idx])

            info_entropy = self._entropy_from_min(idx, self.min_dists[idx])
            info_entropy_array.append(info_entropy)

            layer_cnt = len(universe)
            pruned_cnt = len(prune_indices[idx])
            print(
                f"Conv: {idx} \t  {layer_cnt} --> {layer_cnt - pruned_cnt}\t "
                f"Ratio: {'%.3f' % (100 * pruned_cnt / layer_cnt)}%\t\tLayer Diversity {info_entropy} "
            )

        if len(info_entropy_array) > 0:
            t = torch.tensor(info_entropy_array)
            print(f"\nNetwork Diversity: {t.min().item()}")

        print(f"Total: {self.total}\tNeed to Prune: {self.n_to_prune}\n")
        return prune_indices


class PCCFDMBaronStrategy():
    def __init__(self, convs, ratio):
        self.weights = []
        self.total = 0

        for conv in convs:
            n = len(conv.weight)
            self.weights.append(conv.weight.view(n, -1))
            self.total += n

        self.n_to_prune = int(ratio * self.total)

        self.save_indices = []
        self.dist_matrices = []
        self.max_dists = []
        self.min_dists = []

        print("Init dist matrices")
        self.init_dist_matrices()
        self.init_indices()

        for idx in range(len(self.dist_matrices)):
            self.step_baron(idx)

    @staticmethod
    def max_dist_pair(matrix):
        max_dist = 0.0
        best_pair = ()
        n = len(matrix)
        for i in range(n):
            for j in range(i + 1, n):
                d = matrix[i, j].item()
                if d > max_dist:
                    max_dist = d
                    best_pair = (i, j)
        return best_pair, max_dist

    def init_dist_matrices(self):
        bar_size = 0
        for w in self.weights:
            bar_size += len(w) ** 2

        pbar = tqdm(total=bar_size)
        for w in self.weights:
            n = len(w)
            matrix = torch.zeros([n, n])
            for i in range(n):
                for j in range(n):
                    matrix[i, j] = torch.linalg.norm(w[i] - w[j], ord=2).item()
                    pbar.update(1)
            self.dist_matrices.append(matrix)
        pbar.close()

    def init_indices(self):
        for idx in range(len(self.weights)):
            matrix = self.dist_matrices[idx]
            pair, max_dist = PCCFDMBaronStrategy.max_dist_pair(matrix)

            if not pair:
                self.save_indices.append(set([0]))
                self.max_dists.append(0.0)
                self.min_dists.append(0.0)
            else:
                self.save_indices.append(set(pair))
                self.max_dists.append(max_dist)
                self.min_dists.append(max_dist)

    def pruned_cnt(self):
        total_save = 0
        for s in self.save_indices:
            total_save += len(s)
        return self.total - total_save

    def step_baron(self, idx):
        indices = self.save_indices[idx]
        matrix = self.dist_matrices[idx]

        n = len(matrix)
        print('filter index:', idx, 'n=:', n, 'm=:', len(indices) + 1)
        if len(indices) == n:
            return

        model = pyo.ConcreteModel()
        model.I = pyo.RangeSet(0, n - 1)
        model.z = pyo.Var(model.I, domain=pyo.Binary)
        model.d = pyo.Var(domain=pyo.NonNegativeReals)

        model.obj = pyo.Objective(expr=model.d, sense=pyo.maximize)
        model.num_selected = pyo.Constraint(expr=sum(model.z[i] for i in model.I) == (len(indices) + 1))

        def min_dist_rule(model, i, j):
            if i < j:
                return model.d <= matrix[i, j] + 10 * (1 - model.z[i] * model.z[j])
            else:
                return pyo.Constraint.Skip

        model.min_dist = pyo.Constraint(model.I, model.I, rule=min_dist_rule)

        solver = pyo.SolverFactory('baron')
        solver.options["MaxTime"] = 7200000
        solver.options["EpsR"] = 1e-6

        results = solver.solve(model, tee=True)

        save_point = []
        for i in model.I:
            if pyo.value(model.z[i]) >= 0.99:
                save_point.append(i)

        max_min_dist = pyo.value(model.d)

        self.save_indices[idx] = set(save_point)
        self.min_dists[idx] = max_min_dist

    def _entropy_from_min(self, idx, min_dist):
        denom = self.max_dists[idx]
        return 0.0 if denom == 0 else (min_dist / denom)

    def run(self):
        print("Run")
        L = len(self.weights)

        total_iters = max(0, self.pruned_cnt() - self.n_to_prune)
        pbar = tqdm(total=total_iters)

        while self.pruned_cnt() != self.n_to_prune:
            max_entropy = -1.0
            max_idx = None

            for idx in range(L):
                if len(self.save_indices[idx]) >= len(self.dist_matrices[idx]):
                    continue

                indices = self.save_indices[idx]
                matrix = self.dist_matrices[idx]
                n = len(matrix)

                model = pyo.ConcreteModel()
                model.I = pyo.RangeSet(0, n - 1)
                model.z = pyo.Var(model.I, domain=pyo.Binary)
                model.d = pyo.Var(domain=pyo.NonNegativeReals)

                model.obj = pyo.Objective(expr=model.d, sense=pyo.maximize)
                model.num_selected = pyo.Constraint(expr=sum(model.z[i] for i in model.I) == (len(indices) + 1))

                def min_dist_rule(model, i, j):
                    if i < j:
                        return model.d <= matrix[i, j] + 10 * (1 - model.z[i] * model.z[j])
                    else:
                        return pyo.Constraint.Skip

                model.min_dist = pyo.Constraint(model.I, model.I, rule=min_dist_rule)

                solver = pyo.SolverFactory('baron')
                solver.options["MaxTime"] = 7200000
                solver.options["EpsR"] = 1e-6

                results = solver.solve(model, tee=True)

                next_min_dist = pyo.value(model.d)
                next_entropy = self._entropy_from_min(idx, next_min_dist)

                if next_entropy > max_entropy:
                    max_entropy = next_entropy
                    max_idx = idx

            if max_idx is None:
                print("[Warn] No selectable layer left. Target prune count may be infeasible.")
                break

            self.step_baron(max_idx)
            pbar.update(1)

        pbar.close()

        prune_indices = []
        info_entropy_array = []

        for idx in range(L):
            universe = set(range(len(self.weights[idx])))
            prune_indices.append(universe - self.save_indices[idx])

            info_entropy = self._entropy_from_min(idx, self.min_dists[idx])
            info_entropy_array.append(info_entropy)

            layer_cnt = len(universe)
            pruned_cnt = len(prune_indices[idx])
            print(
                f"Conv: {idx} \t  {layer_cnt} --> {layer_cnt - pruned_cnt}\t "
                f"Ratio: {'%.3f' % (100 * pruned_cnt / layer_cnt)}%\t\tLayer Diversity {info_entropy} "
            )

        if len(info_entropy_array) > 0:
            t = torch.tensor(info_entropy_array)
            print(f"\nNetwork Diversity: {t.min().item()}")

        print(f"Total: {self.total}\tNeed to Prune: {self.n_to_prune}\n")
        return prune_indices


class PCCFDMCPLEXStrategy():
    def __init__(self, convs, ratio):
        self.weights = []
        self.total = 0

        for conv in convs:
            n = len(conv.weight)
            self.weights.append(conv.weight.view(n, -1))
            self.total += n

        self.n_to_prune = int(ratio * self.total)

        self.save_indices = []
        self.dist_matrices = []
        self.max_dists = []
        self.min_dists = []

        print("Init dist matrices")
        self.init_dist_matrices()
        self.init_indices()

        for idx in range(len(self.dist_matrices)):
            self.step_cplex(idx)

    @staticmethod
    def max_dist_pair(matrix):
        max_dist = 0.0
        best_pair = ()
        n = len(matrix)
        for i in range(n):
            for j in range(i + 1, n):
                d = matrix[i, j].item()
                if d > max_dist:
                    max_dist = d
                    best_pair = (i, j)
        return best_pair, max_dist

    def init_dist_matrices(self):
        bar_size = 0
        for w in self.weights:
            bar_size += len(w) ** 2

        pbar = tqdm(total=bar_size)
        for w in self.weights:
            n = len(w)
            matrix = torch.zeros([n, n])
            for i in range(n):
                for j in range(n):
                    matrix[i, j] = torch.linalg.norm(w[i] - w[j], ord=2).item()
                    pbar.update(1)
            self.dist_matrices.append(matrix)
        pbar.close()

    def init_indices(self):
        for idx in range(len(self.weights)):
            matrix = self.dist_matrices[idx]
            pair, max_dist = PCCFDMCPLEXStrategy.max_dist_pair(matrix)

            if not pair:
                self.save_indices.append(set([0]))
                self.max_dists.append(0.0)
                self.min_dists.append(0.0)
            else:
                self.save_indices.append(set(pair))
                self.max_dists.append(max_dist)
                self.min_dists.append(max_dist)

    def pruned_cnt(self):
        total_save = 0
        for s in self.save_indices:
            total_save += len(s)
        return self.total - total_save

    def step_cplex(self, idx):
        indices = self.save_indices[idx]
        matrix = self.dist_matrices[idx]

        n = len(matrix)
        print('filter index:', idx, 'n=:', n, 'm=:', len(indices) + 1)
        if len(indices) == n:
            return

        model = pyo.ConcreteModel()
        model.I = pyo.RangeSet(0, n - 1)
        model.z = pyo.Var(model.I, domain=pyo.Binary)
        model.d = pyo.Var(domain=pyo.NonNegativeReals)

        model.obj = pyo.Objective(expr=model.d, sense=pyo.maximize)
        model.num_selected = pyo.Constraint(expr=sum(model.z[i] for i in model.I) == (len(indices) + 1))

        def min_dist_rule(model, i, j):
            if i < j:
                return model.d <= matrix[i, j] + 10 * (1 - model.z[i]) + 10 * (1 - model.z[j])
            else:
                return pyo.Constraint.Skip

        model.min_dist = pyo.Constraint(model.I, model.I, rule=min_dist_rule)

        solver = pyo.SolverFactory('cplex')
        results = solver.solve(model, tee=True)

        save_point = []
        for i in model.I:
            if pyo.value(model.z[i]) >= 0.99:
                save_point.append(i)

        max_min_dist = pyo.value(model.d)

        self.save_indices[idx] = set(save_point)
        self.min_dists[idx] = max_min_dist

    def _entropy_from_min(self, idx, min_dist):
        denom = self.max_dists[idx]
        return 0.0 if denom == 0 else (min_dist / denom)

    def run(self):
        print("Run")
        L = len(self.weights)

        total_iters = max(0, self.pruned_cnt() - self.n_to_prune)
        pbar = tqdm(total=total_iters)

        while self.pruned_cnt() != self.n_to_prune:
            max_entropy = -1.0
            max_idx = None

            for idx in range(L):
                if len(self.save_indices[idx]) >= len(self.dist_matrices[idx]):
                    continue

                indices = self.save_indices[idx]
                matrix = self.dist_matrices[idx]
                n = len(matrix)

                model = pyo.ConcreteModel()
                model.I = pyo.RangeSet(0, n - 1)
                model.z = pyo.Var(model.I, domain=pyo.Binary)
                model.d = pyo.Var(domain=pyo.NonNegativeReals)

                model.obj = pyo.Objective(expr=model.d, sense=pyo.maximize)
                model.num_selected = pyo.Constraint(expr=sum(model.z[i] for i in model.I) == (len(indices) + 1))

                def min_dist_rule(model, i, j):
                    if i < j:
                        return model.d <= matrix[i, j] + 10 * (1 - model.z[i]) + 10 * (1 - model.z[j])
                    else:
                        return pyo.Constraint.Skip

                model.min_dist = pyo.Constraint(model.I, model.I, rule=min_dist_rule)

                solver = pyo.SolverFactory('cplex')
                results = solver.solve(model, tee=True)

                next_min_dist = pyo.value(model.d)
                next_entropy = self._entropy_from_min(idx, next_min_dist)

                if next_entropy > max_entropy:
                    max_entropy = next_entropy
                    max_idx = idx

            if max_idx is None:
                print("[Warn] No selectable layer left. Target prune count may be infeasible.")
                break

            self.step_cplex(max_idx)
            pbar.update(1)

        pbar.close()

        prune_indices = []
        info_entropy_array = []

        for idx in range(L):
            universe = set(range(len(self.weights[idx])))
            prune_indices.append(universe - self.save_indices[idx])

            info_entropy = self._entropy_from_min(idx, self.min_dists[idx])
            info_entropy_array.append(info_entropy)

            layer_cnt = len(universe)
            pruned_cnt = len(prune_indices[idx])
            print(
                f"Conv: {idx} \t  {layer_cnt} --> {layer_cnt - pruned_cnt}\t "
                f"Ratio: {'%.3f' % (100 * pruned_cnt / layer_cnt)}%\t\tLayer Diversity {info_entropy} "
            )

        if len(info_entropy_array) > 0:
            t = torch.tensor(info_entropy_array)
            print(f"\nNetwork Diversity: {t.min().item()}")

        print(f"Total: {self.total}\tNeed to Prune: {self.n_to_prune}\n")
        return prune_indices

class SUMFDMStrategy():
    def __init__(self, convs, ratio):
        self.weights = []
        self.total = 0
        for conv in convs:
            n = len(conv.weight)
            self.weights.append(conv.weight.view(n, -1))
            self.total += n
        self.n_to_prune = int(ratio * self.total)
        self.save_indices = []
        self.dist_matrices = []
        self.max_dists = []
        self.min_dists = []
        print("Init dist matrices")
        self.init_dist_matrices()
        self.init_indices()


    @staticmethod
    def max_dist_pair(matrix):
        max_dist = 0
        best_pair = ()
        n = len(matrix)
        for i in range(n):
            for j in range(i+1, n):
                if matrix[i, j] > max_dist:
                    max_dist = matrix[i, j].item()
                    best_pair = (i, j)
        return best_pair, max_dist

    def init_dist_matrices(self):
        bar_size = 0
        for w in self.weights:
            bar_size += len(w) ** 2
        pbar = tqdm(total=bar_size)
        for w in self.weights:
            n = len(w)
            matrix = torch.zeros([n, n])
            for i in range(n):
                for j in range(n):
                    matrix[i, j] = torch.linalg.norm(w[i] - w[j], float('2')).item()
                    pbar.update(1)
            self.dist_matrices.append(matrix)
        pbar.close()

    def init_indices(self):
        for idx in range(len(self.weights)):
            matrix = self.dist_matrices[idx]
            max_dist_pair, max_dist = SUMFDMStrategy.max_dist_pair(matrix)
            self.save_indices.append(set(max_dist_pair))
            self.max_dists.append(max_dist)
            self.min_dists.append(max_dist)

    def pruned_cnt(self):
        total_save = 0
        for s in self.save_indices:
            total_save += len(s)
        return self.total - total_save

    # 当前已“保留”的总数
    def saved_cnt(self):
        return sum(len(s) for s in self.save_indices)

    def step(self, idx):
        max_min_dist = 0
        max_min_point = None
        indices = self.save_indices[idx]
        matrix = self.dist_matrices[idx]
        n = len(matrix)
        if len(indices) == n:
            return
        for i in range(n):
            if i in indices:
                continue
            # find min dist between current point and save_points
            cur_min_dist = float("inf")
            for j in indices:
                if matrix[i, j].item() < cur_min_dist:
                    cur_min_dist = matrix[i, j].item()
            # find max min dist point
            if cur_min_dist > max_min_dist:
                max_min_dist = cur_min_dist
                max_min_point = i
        self.save_indices[idx].add(max_min_point)
        self.min_dists[idx] = max_min_dist

    def _peek_next_info(self, idx):
        """
        仅试算：若在第 idx 层按原逻辑（farthest-first）再加 1 个点，
        新的 info_entropy 会是多少，以及下降量是多少。
        不修改任何状态。
        """
        indices = self.save_indices[idx]
        matrix = self.dist_matrices[idx]
        n = len(matrix)
        if len(indices) == n:
            return None  # 已满

        # —— 完全复刻你 step() 里的“层内选点”逻辑（farthest-first）——
        max_min_dist = 0.0
        # max_min_point = None   # 仅试算，不需要真正返回点
        for i in range(n):
            if i in indices:
                continue
            cur_min_dist = float("inf")
            for j in indices:
                dij = matrix[i, j].item()
                if dij < cur_min_dist:
                    cur_min_dist = dij
                    if cur_min_dist == 0.0:
                        break
            if cur_min_dist > max_min_dist:
                max_min_dist = cur_min_dist

        old_ie = self.min_dists[idx] / (self.max_dists[idx] + 1e-12)
        new_ie = max_min_dist          / (self.max_dists[idx] + 1e-12)
        drop  = old_ie - new_ie        # 下降量（越小越好）

        return old_ie, new_ie, drop

    def run(self):
        print("Run")
        filters_cnt = len(self.weights)
        target_keep = self.total - self.n_to_prune
        steps_needed = max(0, target_keep - self.saved_cnt())

        pbar = tqdm(total=steps_needed)
        for _ in range(steps_needed):
            best_layer = None
            best_tuple = None  # (drop, new_ie, remain_capacity)

            # 逐层试算，选“下降最小”的那层
            for idx in range(filters_cnt):
                if len(self.save_indices[idx]) >= len(self.dist_matrices[idx]):
                    continue
                peek = self._peek_next_info(idx)
                if peek is None:
                    continue
                old_ie, new_ie, drop = peek
                remain = len(self.dist_matrices[idx]) - len(self.save_indices[idx])

                # 规则：先比下降量 drop（小优），再比 new_ie（大优），再比剩余可选空间 remain（大优）
                key = (drop, -new_ie, -remain)
                if best_tuple is None or key < best_tuple:
                    best_tuple = key
                    best_layer = idx

            if best_layer is None:
                break

            # 真正“落子”仍然调用你原来的层内逻辑
            self.step(best_layer)
            pbar.update(1)

        pbar.close()

        # —— 以下保持你原来的汇总与返回 —— 
        prune_indices = []
        info_entropy_array = []
        for idx in range(filters_cnt):
            universe = set(range(len(self.weights[idx])))
            prune_indices.append(universe - self.save_indices[idx])
            info_entropy = self.min_dists[idx] / (self.max_dists[idx] + 1e-12)
            info_entropy_array.append(info_entropy)

            layer_cnt = len(universe)
            pruned_cnt = len(prune_indices[idx])
            print(
                f"Conv: {idx} \t  {layer_cnt} --> {layer_cnt - pruned_cnt}\t "
                f"Ratio: {100 * pruned_cnt / max(1, layer_cnt):.3f}%\t\t"
                f"Layer Diversity {info_entropy}"
            )

        ie_t = torch.tensor(info_entropy_array, dtype=torch.float32)
        print(f'\nNetwork Diversity: {ie_t.mean().item()}')
        print(f'Total: {self.total}\tNeed to Prune: {self.n_to_prune}\n')
        return prune_indices


class LNStrategy():
    def __init__(self, p, rev=False):
        self.p = p
        self.rev = rev

    def apply(self, weights, ratio=0.0, round_to=1) -> Sequence[int]:
        if ratio >= 1:
            return []
        n = len(weights)
        l1_norm = torch.norm(weights.view(n, -1), p=self.p, dim=1)
        n_to_prune = int(ratio* n)
        if n_to_prune == 0:
            return []
        threshold = torch.kthvalue(l1_norm, k=n_to_prune).values
        if self.rev:
            indices = torch.nonzero(l1_norm >= threshold).view(-1).tolist()
        else:
            indices = torch.nonzero(l1_norm <= threshold).view(-1).tolist()
        return indices

class RandomStrategy():
    def __init__(self):
        pass
    def apply(self, weights, ratio):
        if ratio >= 1:
            return []
        n = len(weights)
        n_to_prune = int(ratio* n)
        if n_to_prune == 0:
            return []
        universe = [i for i in range(n)]        
        return random.sample(universe, n_to_prune)

class GMStrategy():
    def __init__(self, rev=False) -> None:
        self.rev=rev
    def apply(self, weights, ratio=0.0) -> Sequence[int]:
        if ratio >= 1:
            return []
        n = len(weights)
        w = weights.view(n, -1)
        n_to_prune = int(ratio* n)
        if n_to_prune == 0:
            return []

        # step1: calc center
        center = torch.zeros(w.shape[1], requires_grad=False)
        for index in range(n):
            center = center + w[index]
        center = center / n
        # step2: calc dist(filter, center)
        dists = []
        for index in range(n):
            dists.append((torch.norm(w[index] - center, p='fro'), index))
        dists.sort(key=lambda x:x[0],reverse=self.rev)
        # step3: calc indices
        indices = []
        for i in range(n):
            if i < n_to_prune:
                indices.append(dists[i][1])
        return indices

class kSCStrategy():
    # smallest k-enclosing circle
    def __init__(self) -> None:
        pass
    def apply(self, weights, ratio):
        if ratio >= 1:
            return []
        n = len(weights)
        w = weights.view(n, -1)
        n_to_prune = int(ratio* n)
        if n_to_prune == 0:
            return []
        
        indices = kSC.kSC(n, n - n_to_prune, w)
        
        return indices
        
        