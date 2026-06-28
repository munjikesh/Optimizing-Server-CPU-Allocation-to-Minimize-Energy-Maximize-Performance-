import pulp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

N = 15 #number of users
M = 5#number of servers

alpha = 0.32 #energy constant
w1 = 1 #weight for energy
w2 = 1#weight for performance

C = np.array([600, 300, 700, 500, 500])#server capacities in CPU cores

D = np.array([180, 220, 100, 120, 260, 190, 140, 70, 110, 200, 90, 85, 95, 150, 95])#user demands in CPU cores

d = np.array([
    [1.6, 2.4, 5.3, 3.8, 2.0], [3.9, 2.1, 6.8, 4.1, 3.2], [1.1, 1.5, 2.9, 2.0, 1.3],
    [2.2, 2.6, 4.3, 2.5, 2.1], [5.0, 3.8, 6.2, 5.4, 3.9], [3.2, 2.7, 5.5, 4.0, 2.6],
    [2.6, 2.3, 4.1, 3.4, 1.9], [1.0, 1.3, 2.0, 1.6, 0.9], [2.4, 2.1, 3.6, 2.8, 1.8],
    [3.6, 3.2, 5.9, 4.6, 2.8], [1.8, 1.9, 2.5, 1.9, 1.7], [1.7, 1.6, 2.8, 1.8, 1.4],
    [2.0, 1.9, 3.1, 2.4, 1.6], [2.9, 2.5, 4.6, 3.9, 2.0], [3.3, 2.8, 5.0, 4.2, 2.4]
])#inefficiency score matrix (d_ij)

r = np.array([
    [1.5, 1.7, 0.9, 1.4, 1.2], [2.0, 1.6, 0.7, 2.2, 1.3], [1.6, 1.4, 1.1, 1.5, 1.2],
    [1.4, 1.8, 1.0, 1.6, 1.1], [2.2, 1.9, 0.6, 2.4, 1.5], [1.3, 1.6, 0.9, 1.5, 1.1],
    [1.2, 1.1, 1.0, 1.3, 1.0], [2.6, 1.3, 1.1, 1.8, 1.4], [1.5, 1.4, 1.0, 1.3, 1.2],
    [1.9, 1.7, 0.8, 2.1, 1.4], [1.4, 1.5, 1.1, 1.2, 1.3], [1.3, 1.25,1.05,1.1, 1.0],
    [1.45,1.35,1.1, 1.25,1.05], [1.6, 1.5, 0.95,1.45,1.2], [1.8, 1.6, 0.85,1.7, 1.25]
])#performance score matrix (r_ij)

server_names = [f"Server {j}" for j in range(M)]
server_names_short = [f"S{j}" for j in range(M)]
app_names = [f"Application {i}" for i in range(N)]

P_linear = (w1 * alpha * (d ** 2)) - (w2 * r)
#x is the 2D vector matrix
def check_server_capacity(x):#this function checks if server capacities are not exceeded
    for j in range(M):#going through every server j
        if np.sum(x[:, j]) > C[j] + 0.01: return False #0.01 is to avoid floating point issues
    return True
#floating point issues occurs because machine user cannot represent some decimal numbers exactly as they are using in base 2
def check_user_requirements(x):#this function checks if user demands are met
    for i in range(N):#go though all user i
        if np.sum(x[i, :]) < D[i] - 0.01: return False
    return True

def calculate_metrics_linear(x):#function to calculate energy, performence and cost
    energy = np.sum(w1 * alpha * (d ** 2) * x)
    throughput = np.sum(w2 * r * x)
    cost = energy - throughput
    return energy, throughput, cost

def solve_optimal_lp():#this function builds,solves and evaluates the optimal LP solution using PuLP
    print("\n" + "=" * 80)
    print("PHASE 1 - METHOD 1: OPTIMAL LP SOLUTION (PuLP)")
    print("=" * 80)
    
    prob = pulp.LpProblem("Server_CPU_Allocation_LP", pulp.LpMinimize)#creates a new lp problem instance named "Server_CPU_Allocation_LP" with the objective to minimize
    #next we will define the decision variables
    x = pulp.LpVariable.dicts("x", (range(N), range(M)), lowBound=0)
    #creates a dictionary for 2D x decision variables with lower bound 0 for each user i and server j   
    prob += pulp.lpSum([P_linear[i, j] * x[i][j] for i in range(N) for j in range(M)]), "Total_Net_Cost"#builds a lp objective fn 
    for j in range(M):
        prob += pulp.lpSum([x[i][j] for i in range(N)]) <= C[j], f"Server_Capacity_{j}"#for server capacity constraints
    for i in range(N):
        prob += pulp.lpSum([x[i][j] for j in range(M)]) >= D[i], f"User_Requirement_{i}"#for user demand constraints
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))#solve the lp using the default solver CBC with no output messages
    
    x_optimal = np.zeros((N, M))#get the optimal x 2D allocation matrix
    for i in range(N):
        for j in range(M):
            x_optimal[i, j] = x[i][j].varValue
    
    energy, throughput, cost = calculate_metrics_linear(x_optimal)# compute energy, throughput and cost for the optimal solution
    
    print(f"\nStatus: {pulp.LpStatus[prob.status]}")
    print(f"\nResults:")
    print(f"  Total Energy Cost: {energy:.2f}")
    print(f"  Total Throughput: {throughput:.2f}")
    print(f"  Objective Cost (Z): {cost:.2f}")
    print(f"\nConstraint Verification:")
    print(f"  All users meet minimum requirement: {check_user_requirements(x_optimal)}")
    print(f"  All servers within capacity: {check_server_capacity(x_optimal)}")
    
    print_allocation_details(x_optimal, "Optimal LP")
    return x_optimal, energy, throughput, cost

def ensure_constraints(x):#this function adjusts allocation(x) to ensure constraints are met
    x_fixed = x.copy()#to avoid modifying the original x
    for j in range(M):#looping through servers
        total_load = np.sum(x_fixed[:, j])#total capacity load on server j
        if total_load > C[j]:#if load exceeds capacity
            scale_factor = C[j] / total_load#to calculate how much we have to scale down
            x_fixed[:, j] *= scale_factor
    
    for i in range(N):#fixing user demands constraint
        current_allocation = np.sum(x_fixed[i, :])#current allocation for user i
        if current_allocation < D[i] - 1e-6:#if demands are not met
            deficit = D[i] - current_allocation#how much more cpu resource is needed
            for j in range(M):#going through servers to fill the deficit
                server_spare = C[j] - np.sum(x_fixed[:, j])#to see how much spare capacity server j has
                if server_spare > 0:
                    additional = min(deficit, server_spare)
                    x_fixed[i, j] += additional
                    deficit -= additional
                    if deficit <= 1e-6: break
    return x_fixed
#we could also solve this using a small LP to minimize the total adjustment needed and also we could use penalty methods...

def solve_round_robin():#every user gets equal share from each server
    print("\n" + "=" * 80)
    print("PHASE 1 - METHOD 2: ROUND ROBIN BASELINE")
    print("=" * 80)
    
    x_rr = np.zeros((N, M))#to create the allocation matrix
    for i in range(N):#to distribute user i's demand equally across all servers
        allocation_per_server = D[i] / M
        for j in range(M):
            x_rr[i, j] = allocation_per_server
    
    x_rr = ensure_constraints(x_rr)#while equally distributing we may overload some servers so here we are going to use the ensure_constraints function to fix that
    energy, throughput, cost = calculate_metrics_linear(x_rr)#compute energy, throughput and cost for the round robin solution
    
    print(f"\nResults:")
    print(f"  Total Energy Cost: {energy:.2f}")
    print(f"  Total Throughput: {throughput:.2f}")
    print(f"  Objective Cost (Z): {cost:.2f}")
    print(f"\nConstraint Verification:")
    print(f"  All users meet minimum requirement: {check_user_requirements(x_rr)}")
    print(f"  All servers within capacity: {check_server_capacity(x_rr)}")
    
    return x_rr, energy, throughput, cost

def solve_random():#this function creates a random allocation baseline
    print("\n" + "=" * 80)
    print("PHASE 1 - METHOD 3: RANDOM BASELINE")
    print("=" * 80)
    
    x_rand = np.zeros((N, M))
    for i in range(N):
        weights = np.random.random(M)#for each user i, generate random weights for each server j
        weights /= weights.sum()
        for j in range(M):
            x_rand[i, j] = D[i] * weights[j]
    
    x_rand = ensure_constraints(x_rand)#fix violations of constraints
    energy, throughput, cost = calculate_metrics_linear(x_rand)#compute energy, throughput and cost for the random solution
    
    print(f"\nResults:")
    print(f"  Total Energy Cost: {energy:.2f}")
    print(f"  Total Throughput: {throughput:.2f}")
    print(f"  Objective Cost (Z): {cost:.2f}")
    print(f"\nConstraint Verification:")
    print(f"  All users meet minimum requirement: {check_user_requirements(x_rand)}")
    print(f"  All servers within capacity: {check_server_capacity(x_rand)}")
    
    return x_rand, energy, throughput, cost

def solve_greedy_performance():#this function implements a greedy allocation method focusing on performance first
    print("\n" + "=" * 80)
    print("PHASE 1 - METHOD 4: GREEDY (PERFORMANCE FIRST)")
    print("=" * 80)

    all_pairs = []
    for i in range(N):
        for j in range(M):
            all_pairs.append((r[i, j], i, j))#builds a list of tuples (performance score, user index, server index)
    all_pairs.sort(key=lambda x: x[0], reverse=True)#sort in dec order wrt performance score
    
    demand_remaining = D.copy()#how much more cpu user i needs
    capacity_remaining = C.copy()#how much more cpu server j can provide
    x_plan = np.zeros((N, M))#creating an allocation matrix
    
    for (r_val, i, j) in all_pairs:
        need = demand_remaining[i]
        if need <= 1e-6: continue#already filled user i's demand
        available = capacity_remaining[j]
        if available <= 1e-6: continue#server j is full
            
        alloc = min(need, available)#put cpu as much as possible to best performance server
        x_plan[i, j] = alloc
        demand_remaining[i] -= alloc#updation
        capacity_remaining[j] -= alloc
    
    x_plan = ensure_constraints(x_plan)#fix violations of constraints
    energy, throughput, cost = calculate_metrics_linear(x_plan)#compute energy, throughput and cost for the greedy performance solution
    
    print(f"\nResults:")
    print(f"  (This method ignores energy (d_ij) and only chases performance (r_ij))")
    print(f"  Total Energy Cost: {energy:.2f}")
    print(f"  Total Throughput: {throughput:.2f}")
    print(f"  Objective Cost (Z): {cost:.2f}")
    print(f"\nConstraint Verification:")
    print(f"  All users meet minimum requirement: {check_user_requirements(x_plan)}")
    print(f"  All servers within capacity: {check_server_capacity(x_plan)}")
    
    return x_plan, energy, throughput, cost

def print_allocation_details(x, method_name):
    print(f"\n--- Allocation Details for: {method_name} ---")
    print(f"\n{'Application':<30} {'Total CPU':<12} Server Distribution")
    print("-" * 80)
    for i in range(N):
        total = np.sum(x[i, :])
        dist = " | ".join([f"{server_names_short[j]}: {x[i,j]:.1f}" for j in range(M)])
        print(f"  {app_names[i]:<28}: {total:>6.1f} cores  [{dist}]")
    
    print(f"\n{'Server Load Summary':<30}")
    print("-" * 50)
    for j in range(M):
        load = np.sum(x[:, j])
        util = (load / C[j]) * 100
        print(f"  {server_names[j]:<25}: {load:>6.1f} / {C[j]} cores ({util:>5.1f}% utilized)")

def main_lp():
    print("=" * 80)
    print("ENERGY-MINIMIZED SERVER CPU ALLOCATION - PHASE 1 (LP)")
    print("=" * 80)
    
    lp_results = {}
    x_opt_lp, e_opt, t_opt, c_opt = solve_optimal_lp()
    lp_results['Optimal LP'] = {'allocation': x_opt_lp, 'energy': e_opt, 'throughput': t_opt, 'cost': c_opt}
    
    x_rr, e_rr, t_rr, c_rr = solve_round_robin()
    lp_results['Round Robin'] = {'allocation': x_rr, 'energy': e_rr, 'throughput': t_rr, 'cost': c_rr}
    
    x_rand, e_rand, t_rand, c_rand = solve_random()
    lp_results['Random'] = {'allocation': x_rand, 'energy': e_rand, 'throughput': t_rand, 'cost': c_rand}

    x_greedy, e_greedy, t_greedy, c_greedy = solve_greedy_performance()
    lp_results['Greedy (Perf. First)'] = {'allocation': x_greedy, 'energy': e_greedy, 'throughput': t_greedy, 'cost': c_greedy}
    
    print("\n" + "=" * 80)
    print("PHASE 1: COMPARATIVE ANALYSIS (LINEAR MODEL)")
    print("=" * 80)
    lp_data = []
    for method, metrics in lp_results.items():
        lp_data.append({'Method': method, 'Energy Cost': metrics['energy'], 'Throughput': metrics['throughput'], 'Objective (Z)': metrics['cost']})
    df_lp = pd.DataFrame(lp_data).set_index('Method')
    df_lp_print = df_lp.copy()
    df_lp_print['Energy Cost'] = df_lp_print['Energy Cost'].map('{:,.2f}'.format)
    df_lp_print['Throughput'] = df_lp_print['Throughput'].map('{:,.2f}'.format)
    df_lp_print['Objective (Z)'] = df_lp_print['Objective (Z)'].map('{:,.2f}'.format)
    print("\n" + df_lp_print.to_string())
    
    optimal_lp_cost = lp_results['Optimal LP']['cost']
    print("\n" + "-" * 80)
    print("IMPROVEMENT ANALYSIS (vs Optimal LP)")
    print("-" * 80)
    if abs(optimal_lp_cost) > 1e-6:
        for method in ['Round Robin', 'Random', 'Greedy (Perf. First)']:
            cost = lp_results[method]['cost']
            improvement = ((cost - optimal_lp_cost) / abs(cost) * 100)
            print(f"Optimal LP is better than {method} by: {improvement:.2f}%")
    print(f"Optimal LP achieves: Lowest objective cost (Z = {optimal_lp_cost:.2f})")

    print("\n" + "=" * 80)
    print("GENERATING PHASE 1 VISUALIZATIONS...")
    print("=" * 80)
    
    methods_lp = list(lp_results.keys())
    costs_lp = [lp_results[m]['cost'] for m in methods_lp]
    
    plt.figure(figsize=(10, 6))
    colors_lp = ['#4CAF50', '#FFC107', '#F44336', '#2196F3']
    bars = plt.bar(methods_lp, costs_lp, color=colors_lp, alpha=0.8)
    plt.title('Phase 1: Linear Model (LP) Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Objective Cost (Z) - Lower is Better', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center')
    plt.show()

    plt.figure(figsize=(8, 8))
    for method in lp_results:
        e = lp_results[method]['energy']
        t = lp_results[method]['throughput']
        plt.scatter(e, t, s=200, label=method, alpha=0.7)
        plt.text(e, t + 20, method, fontsize=9)
    plt.xlabel("Total Energy Cost", fontsize=12)
    plt.ylabel("Total Throughput", fontsize=12)
    plt.title("Phase 1: Energy vs. Throughput Trade-off", fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(x_opt_lp, cmap="YlGnBu", annot=True, fmt=".1f", xticklabels=server_names_short, yticklabels=app_names)
    plt.title("Optimal LP Allocation Heatmap", fontsize=16, fontweight='bold')
    plt.xlabel("Servers", fontsize=12)
    plt.ylabel("Applications", fontsize=12)
    plt.show()


if __name__ == "__main__":
    main_lp()
