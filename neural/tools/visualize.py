"""
Visualization tools for the artificial brain.
Shows brain activity, connectivity, and dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from ..engines.artificial_brain import ArtificialBrain


def plot_brain_activity(brain: ArtificialBrain, duration_ms: float = 100.0):
    """Plot brain activity over time"""
    time_points = []
    region_activities = {name: [] for name in brain.regions.keys()}
    spike_counts = {name: [] for name in brain.regions.keys()}
    
    dt = brain.simulation_dt
    num_steps = int(duration_ms / dt)
    
    initial_spikes = {
        name: sum(n.state.spike_count for n in region.neurons)
        for name, region in brain.regions.items()
    }
    
    for step in range(num_steps):
        brain.step()
        time_points.append(brain.simulation_time)
        
        for name, region in brain.regions.items():
            region_activities[name].append(region.get_activity())
            current_spikes = sum(n.state.spike_count for n in region.neurons)
            spike_counts[name].append(current_spikes - initial_spikes[name])
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1 = axes[0]
    for name, activities in region_activities.items():
        ax1.plot(time_points, activities, label=name, linewidth=2)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Average Membrane Potential (mV)')
    ax1.set_title('Brain Region Activity Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    for name, spikes in spike_counts.items():
        ax2.plot(time_points, spikes, label=name, linewidth=2)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Cumulative Spike Count')
    ax2.set_title('Spike Activity Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_neurotransmitter_levels(brain: ArtificialBrain, duration_ms: float = 100.0):
    """Plot neurotransmitter levels over time"""
    time_points = []
    nt_levels = {nt: [] for nt in brain.neurotransmitters.levels.keys()}
    
    dt = brain.simulation_dt
    num_steps = int(duration_ms / dt)
    
    for step in range(num_steps):
        brain.step()
        time_points.append(brain.simulation_time)
        
        for nt, level in brain.neurotransmitters.levels.items():
            nt_levels[nt].append(level)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for nt, levels in nt_levels.items():
        ax.plot(time_points, levels, label=nt, linewidth=2)
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neurotransmitter Level')
    ax.set_title('Neurotransmitter Levels Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_connectivity_matrix(brain: ArtificialBrain):
    """Plot connectivity between brain regions"""
    region_names = list(brain.regions.keys())
    n_regions = len(region_names)
    
    connectivity = np.zeros((n_regions, n_regions))
    
    for i, (name1, region1) in enumerate(brain.regions.items()):
        for j, (name2, region2) in enumerate(brain.regions.items()):
            if i == j:
                connectivity[i, j] = len(region1.synapses)
            else:
                count = 0
                for synapse in region1.synapses:
                    if synapse.post_neuron in region2.neurons:
                        count += 1
                connectivity[i, j] = count
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(connectivity, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n_regions))
    ax.set_yticks(range(n_regions))
    ax.set_xticklabels(region_names, rotation=45, ha='right')
    ax.set_yticklabels(region_names)
    ax.set_title('Inter-Region Connectivity Matrix')
    
    for i in range(n_regions):
        for j in range(n_regions):
            text = ax.text(j, i, int(connectivity[i, j]),
                          ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax, label='Number of Synapses')
    plt.tight_layout()
    return fig


def plot_memory_traces(brain: ArtificialBrain, duration_ms: float = 200.0):
    """Plot memory formation and consolidation"""
    time_points = []
    working_mem_counts = []
    longterm_mem_counts = []
    working_strengths = []
    longterm_strengths = []
    
    dt = brain.simulation_dt
    num_steps = int(duration_ms / dt)
    
    for step in range(num_steps):
        brain.step()
        time_points.append(brain.simulation_time)
        
        stats = brain.memory.get_memory_stats()
        working_mem_counts.append(stats['working_memory_count'])
        longterm_mem_counts.append(stats['long_term_memory_count'])
        working_strengths.append(stats['avg_working_strength'])
        longterm_strengths.append(stats['avg_longterm_strength'])
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1 = axes[0]
    ax1.plot(time_points, working_mem_counts, label='Working Memory', linewidth=2, marker='o', markersize=4)
    ax1.plot(time_points, longterm_mem_counts, label='Long-term Memory', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Number of Memory Traces')
    ax1.set_title('Memory Formation Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(time_points, working_strengths, label='Working Memory Strength', linewidth=2, marker='o', markersize=4)
    ax2.plot(time_points, longterm_strengths, label='Long-term Memory Strength', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Average Memory Strength')
    ax2.set_title('Memory Strength Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_brain_state(brain: ArtificialBrain):
    """Create a comprehensive visualization of current brain state"""
    fig = plt.figure(figsize=(16, 10))
    
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    region_names = list(brain.regions.keys())
    activities = [brain.regions[name].get_activity() for name in region_names]
    ax1.bar(region_names, activities, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_ylabel('Activity (mV)')
    ax1.set_title('Region Activities')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2 = fig.add_subplot(gs[0, 1])
    spike_rates = [brain.regions[name].get_spike_rate() for name in region_names]
    ax2.bar(region_names, spike_rates, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_ylabel('Spike Rate (Hz)')
    ax2.set_title('Spike Rates')
    ax2.tick_params(axis='x', rotation=45)
    
    ax3 = fig.add_subplot(gs[0, 2])
    nt_levels = brain.neurotransmitters.get_state()
    ax3.barh(list(nt_levels.keys()), list(nt_levels.values()), color='#95E1D3')
    ax3.set_xlabel('Level')
    ax3.set_title('Neurotransmitter Levels')
    
    ax4 = fig.add_subplot(gs[1, :2])
    mem_stats = brain.memory.get_memory_stats()
    categories = ['Working\nMemory', 'Long-term\nMemory', 'Total\nPatterns']
    values = [
        mem_stats['working_memory_count'],
        mem_stats['long_term_memory_count'],
        mem_stats['total_patterns']
    ]
    ax4.bar(categories, values, color=['#F38181', '#AA96DA', '#FCBAD3'])
    ax4.set_ylabel('Count')
    ax4.set_title('Memory Statistics')
    
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    summary_text = f"""
    Brain Summary
    
    Total Neurons: {brain.get_total_neurons()}
    Total Synapses: {brain.get_total_synapses()}
    Simulation Time: {brain.simulation_time:.1f} ms
    
    Regions:
    """
    for name, region in brain.regions.items():
        summary_text += f"  {name}: {len(region.neurons)} neurons\n"
    
    ax5.text(0.1, 0.5, summary_text, fontsize=10, 
             verticalalignment='center', family='monospace')
    
    ax6 = fig.add_subplot(gs[2, :])
    plot_connectivity_matrix_in_axes(brain, ax6)
    
    plt.suptitle('Artificial Brain State Visualization', fontsize=16, y=0.98)
    return fig


def plot_connectivity_matrix_in_axes(brain: ArtificialBrain, ax):
    """Helper to plot connectivity in existing axes"""
    region_names = list(brain.regions.keys())
    n_regions = len(region_names)
    
    connectivity = np.zeros((n_regions, n_regions))
    
    for i, (name1, region1) in enumerate(brain.regions.items()):
        for j, (name2, region2) in enumerate(brain.regions.items()):
            if i == j:
                connectivity[i, j] = len(region1.synapses)
            else:
                count = 0
                for synapse in region1.synapses:
                    if synapse.post_neuron in region2.neurons:
                        count += 1
                connectivity[i, j] = count
    
    im = ax.imshow(connectivity, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n_regions))
    ax.set_yticks(range(n_regions))
    ax.set_xticklabels(region_names, rotation=45, ha='right')
    ax.set_yticklabels(region_names)
    ax.set_title('Inter-Region Connectivity Matrix')
    
    for i in range(n_regions):
        for j in range(n_regions):
            text = ax.text(j, i, int(connectivity[i, j]),
                          ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax, label='Number of Synapses')

