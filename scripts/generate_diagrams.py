"""
Diagram Generator for AegisAI Presentations
Creates publication-quality architecture and flow diagrams
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
from pathlib import Path


# Color scheme
COLORS = {
    'perception': '#4A90D9',
    'reasoning': '#50B86E',
    'actuation': '#E8854C',
    'communication': '#9B59B6',
    'background': '#F5F5F5',
    'text': '#2C3E50',
    'highlight': '#E74C3C',
    'water': '#3498DB',
    'solar': '#F1C40F',
}


def create_system_architecture():
    """Create the main system architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')
    
    # Title
    ax.text(7, 9.5, 'AegisAI Buoy System Architecture', 
            fontsize=18, fontweight='bold', ha='center', color=COLORS['text'])
    
    # Main container
    main_box = FancyBboxPatch((0.5, 0.5), 13, 8.5, 
                               boxstyle="round,pad=0.05,rounding_size=0.3",
                               facecolor=COLORS['background'], edgecolor=COLORS['text'],
                               linewidth=2)
    ax.add_patch(main_box)
    
    # Section labels
    ax.text(2.5, 8.5, 'PERCEPTION', fontsize=12, fontweight='bold', 
            ha='center', color=COLORS['perception'])
    ax.text(7, 8.5, 'REASONING', fontsize=12, fontweight='bold', 
            ha='center', color=COLORS['reasoning'])
    ax.text(11.5, 8.5, 'ACTUATION', fontsize=12, fontweight='bold', 
            ha='center', color=COLORS['actuation'])
    
    # Perception modules
    modules = [
        (1.5, 6.5, 2, 1.2, 'Water Sensors', 'pH, Temp, Turb, EC', COLORS['perception']),
        (1.5, 4.5, 2, 1.2, 'Camera', '(Optional)', COLORS['perception']),
        (1.5, 2.5, 2, 1.2, 'GPS + Battery', 'Power Monitor', COLORS['perception']),
        (5.5, 6.5, 2.5, 1.2, 'Anomaly Detector', 'IF + Autoencoder', COLORS['reasoning']),
        (5.5, 4.5, 2.5, 1.2, 'LSTM Predictor', 'Temporal Forecast', COLORS['reasoning']),
        (5.5, 2.5, 2.5, 1.2, 'PPO Agent', 'Decision Making', COLORS['reasoning']),
        (10, 6.5, 2.5, 1.2, 'Sample Pump', 'Peristaltic', COLORS['actuation']),
        (10, 4.5, 2.5, 1.2, 'LoRa Radio', 'Fleet Mesh', COLORS['actuation']),
        (10, 2.5, 2.5, 1.2, 'ESP32-S3', 'MCU Control', COLORS['actuation']),
    ]
    
    for x, y, w, h, label, sublabel, color in modules:
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor='white', edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.65, label, fontsize=10, fontweight='bold',
                ha='center', va='center', color=COLORS['text'])
        ax.text(x + w/2, y + h*0.3, sublabel, fontsize=8,
                ha='center', va='center', color='gray')
    
    # Arrows
    arrows = [
        (3.5, 7.1, 5.3, 7.1),
        (3.5, 5.1, 5.3, 5.1),
        (3.5, 3.1, 5.3, 3.1),
        (8.0, 7.1, 9.8, 7.1),
        (8.0, 5.1, 9.8, 5.1),
        (8.0, 3.1, 9.8, 3.1),
        (6.75, 6.5, 6.75, 5.7),
        (6.75, 4.5, 6.75, 3.7),
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=1.5))
    
    # Fleet coordination box
    fleet_box = FancyBboxPatch((4.5, 0.8), 5, 1.2,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor='white', edgecolor=COLORS['communication'],
                               linewidth=2, linestyle='--')
    ax.add_patch(fleet_box)
    ax.text(7, 1.4, 'Fleet Coordination (MAPPO)', fontsize=10, fontweight='bold',
            ha='center', va='center', color=COLORS['communication'])
    
    output_path = Path('docs/diagrams/system_architecture.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_anomaly_cascade():
    """Create the anomaly detection cascade diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(6, 7.5, 'Cascade Anomaly Detection', 
            fontsize=16, fontweight='bold', ha='center', color=COLORS['text'])
    
    # Input
    input_box = FancyBboxPatch((0.5, 5.5), 2, 1,
                               boxstyle="round,pad=0.02", facecolor=COLORS['water'],
                               edgecolor=COLORS['text'], linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 6, 'Sensor\nReadings', fontsize=10, ha='center', va='center', 
            color='white', fontweight='bold')
    
    # Isolation Forest
    if_box = FancyBboxPatch((4, 5.5), 2.5, 1,
                            boxstyle="round,pad=0.02", facecolor=COLORS['reasoning'],
                            edgecolor=COLORS['text'], linewidth=2)
    ax.add_patch(if_box)
    ax.text(5.25, 6, 'Isolation\nForest', fontsize=10, ha='center', va='center',
            color='white', fontweight='bold')
    
    ax.annotate('', xy=(4, 6), xytext=(2.5, 6),
               arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=2))
    
    # Low score (normal)
    ax.annotate('', xy=(3.5, 3), xytext=(5.25, 5.5),
               arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(3.5, 4.3, 'Score < 0.3', fontsize=9, color='green', fontweight='bold')
    
    normal_box = FancyBboxPatch((2.5, 2), 2, 1,
                                boxstyle="round,pad=0.02", facecolor='#27AE60',
                                edgecolor=COLORS['text'], linewidth=2)
    ax.add_patch(normal_box)
    ax.text(3.5, 2.5, 'NORMAL\n(Skip AE)', fontsize=10, ha='center', va='center',
            color='white', fontweight='bold')
    
    # High score (anomaly)
    ax.annotate('', xy=(8.5, 3), xytext=(5.25, 5.5),
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(7.5, 4.3, 'Score > 0.7', fontsize=9, color='red', fontweight='bold')
    
    anomaly_box = FancyBboxPatch((7.5, 2), 2, 1,
                                 boxstyle="round,pad=0.02", facecolor=COLORS['highlight'],
                                 edgecolor=COLORS['text'], linewidth=2)
    ax.add_patch(anomaly_box)
    ax.text(8.5, 2.5, 'ANOMALY\n(Sample!)', fontsize=10, ha='center', va='center',
            color='white', fontweight='bold')
    
    # Middle score (uncertain)
    ax.annotate('', xy=(5.25, 3.5), xytext=(5.25, 5.5),
               arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    ax.text(6.5, 4.5, '0.3 - 0.7', fontsize=9, color='orange', fontweight='bold')
    
    ae_box = FancyBboxPatch((4, 2.5), 2.5, 1,
                            boxstyle="round,pad=0.02", facecolor='#F39C12',
                            edgecolor=COLORS['text'], linewidth=2)
    ax.add_patch(ae_box)
    ax.text(5.25, 3, 'Autoencoder\n(Deep Check)', fontsize=10, ha='center', va='center',
            color='white', fontweight='bold')
    
    ax.annotate('', xy=(5.25, 1), xytext=(5.25, 2.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=2))
    
    final_box = FancyBboxPatch((4, 0.3), 2.5, 0.7,
                               boxstyle="round,pad=0.02", facecolor=COLORS['text'],
                               edgecolor=COLORS['text'], linewidth=2)
    ax.add_patch(final_box)
    ax.text(5.25, 0.65, 'Combined Score', fontsize=10, ha='center', va='center',
            color='white', fontweight='bold')
    
    ax.text(10, 6, '60% compute\nsavings', fontsize=11, ha='center', va='center',
            color=COLORS['reasoning'], fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['reasoning']))
    
    output_path = Path('docs/diagrams/anomaly_cascade.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_fleet_coordination():
    """Create fleet coordination diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(6, 9.5, 'Multi-Buoy Fleet Coordination', 
            fontsize=16, fontweight='bold', ha='center', color=COLORS['text'])
    
    # River background
    river = FancyBboxPatch((0.5, 1), 11, 7,
                           boxstyle="round,pad=0.1",
                           facecolor='#D4E6F1', edgecolor=COLORS['water'],
                           linewidth=3)
    ax.add_patch(river)
    ax.text(6, 0.5, 'River / Lake Coverage Area', fontsize=11, ha='center',
            color=COLORS['water'], fontstyle='italic')
    
    # Buoys
    buoy_positions = [(2, 6), (4.5, 3), (7.5, 5.5), (10, 3.5)]
    buoy_colors = [COLORS['perception'], COLORS['perception'], 
                   COLORS['highlight'], COLORS['perception']]
    buoy_labels = ['Buoy 1', 'Buoy 2', 'Buoy 3\n(Anomaly!)', 'Buoy 4']
    
    for i, ((x, y), color, label) in enumerate(zip(buoy_positions, buoy_colors, buoy_labels)):
        circle = Circle((x, y), 0.5, facecolor=color, edgecolor='white', linewidth=3)
        ax.add_patch(circle)
        ax.text(x, y-0.9, label, fontsize=9, ha='center', fontweight='bold',
                color=COLORS['text'])
        
        coverage = Circle((x, y), 1.8, facecolor='none', edgecolor=color,
                          linewidth=1, linestyle='--', alpha=0.5)
        ax.add_patch(coverage)
    
    # LoRa communication lines
    for i, (x1, y1) in enumerate(buoy_positions):
        for j, (x2, y2) in enumerate(buoy_positions):
            if i < j:
                ax.plot([x1, x2], [y1, y2], 'k--', alpha=0.3, linewidth=1)
    
    # Convergence arrows
    ax.annotate('', xy=(6.5, 5.5), xytext=(4.5, 3),
               arrowprops=dict(arrowstyle='->', color=COLORS['highlight'], lw=2))
    ax.annotate('', xy=(8.5, 5), xytext=(10, 3.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['highlight'], lw=2))
    
    ax.text(11.5, 8.5, 'Convergence\nbehavior', fontsize=9, ha='center',
            color=COLORS['highlight'], fontweight='bold')
    
    output_path = Path('docs/diagrams/fleet_coordination.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_power_flow():
    """Create power management flow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(5, 7.5, 'Adaptive Power Management', 
            fontsize=16, fontweight='bold', ha='center', color=COLORS['text'])
    
    # Solar panel
    solar = FancyBboxPatch((0.5, 5.5), 2, 1.2,
                           boxstyle="round,pad=0.02", facecolor=COLORS['solar'],
                           edgecolor=COLORS['text'], linewidth=2)
    ax.add_patch(solar)
    ax.text(1.5, 6.1, 'Solar Panel\n5W peak', fontsize=10, ha='center', 
            va='center', fontweight='bold')
    
    # Battery
    battery = FancyBboxPatch((4, 5.5), 2, 1.2,
                             boxstyle="round,pad=0.02", facecolor='#95A5A6',
                             edgecolor=COLORS['text'], linewidth=2)
    ax.add_patch(battery)
    ax.text(5, 6.1, 'Battery\n20 Wh', fontsize=10, ha='center', 
            va='center', fontweight='bold', color='white')
    
    ax.annotate('', xy=(4, 6.1), xytext=(2.5, 6.1),
               arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=2))
    
    # PPO controller
    ppo = FancyBboxPatch((7.5, 5.5), 2, 1.2,
                         boxstyle="round,pad=0.02", facecolor=COLORS['reasoning'],
                         edgecolor=COLORS['text'], linewidth=2)
    ax.add_patch(ppo)
    ax.text(8.5, 6.1, 'PPO Agent\n(Power Aware)', fontsize=10, ha='center', 
            va='center', fontweight='bold', color='white')
    
    ax.annotate('', xy=(7.5, 6.1), xytext=(6, 6.1),
               arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=2))
    ax.text(6.75, 6.5, 'Level', fontsize=8, ha='center', color='gray')
    
    # Power modes
    modes = [
        (1.5, 2.5, 'ACTIVE', '600s interval\n0.5W avg', COLORS['perception']),
        (5, 2.5, 'ALERT', '30s interval\n1W avg', COLORS['highlight']),
        (8.5, 2.5, 'LOW POWER', '3600s interval\n0.1W avg', '#95A5A6'),
    ]
    
    for x, y, label, desc, color in modes:
        box = FancyBboxPatch((x-1, y-0.7), 2, 1.4,
                             boxstyle="round,pad=0.02", facecolor=color,
                             edgecolor=COLORS['text'], linewidth=2)
        ax.add_patch(box)
        ax.text(x, y+0.2, label, fontsize=10, ha='center', va='center',
                fontweight='bold', color='white')
        ax.text(x, y-0.3, desc, fontsize=8, ha='center', va='center', color='white')
    
    for x, _, _, _, _ in modes:
        ax.annotate('', xy=(x, 3.9), xytext=(8.5, 5.5),
                   arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=1.5))
    
    ax.text(5, 4.5, 'Dynamic Mode Selection', fontsize=10, ha='center',
            fontstyle='italic', color='gray')
    
    output_path = Path('docs/diagrams/power_management.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_data_flow():
    """Create the main data flow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    ax.text(7, 5.5, 'Inference Pipeline (15ms total)', 
            fontsize=16, fontweight='bold', ha='center', color=COLORS['text'])
    
    stages = [
        (1, 2.5, 'Sensor\nRead', '10ms', COLORS['perception']),
        (3.5, 2.5, 'Anomaly\nDetection', '1ms', COLORS['reasoning']),
        (6, 2.5, 'LSTM\nPrediction', '3ms', COLORS['reasoning']),
        (8.5, 2.5, 'PPO\nDecision', '1ms', COLORS['reasoning']),
        (11, 2.5, 'Action\nExecution', '~0ms', COLORS['actuation']),
    ]
    
    for i, (x, y, label, time, color) in enumerate(stages):
        box = FancyBboxPatch((x-0.8, y-0.7), 1.6, 1.4,
                             boxstyle="round,pad=0.02", facecolor=color,
                             edgecolor=COLORS['text'], linewidth=2)
        ax.add_patch(box)
        ax.text(x, y+0.2, label, fontsize=10, ha='center', va='center',
                fontweight='bold', color='white')
        ax.text(x, y-0.4, time, fontsize=9, ha='center', va='center',
                color='white', fontstyle='italic')
        
        if i < len(stages) - 1:
            ax.annotate('', xy=(stages[i+1][0]-0.9, y), xytext=(x+0.9, y),
                       arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=2))
    
    actions = [
        (11, 0.8, 'Sample'),
        (12.5, 0.8, 'Alert'),
        (11.75, 1.5, 'Wait'),
    ]
    
    for x, y, label in actions:
        ax.text(x, y, label, fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    
    output_path = Path('docs/diagrams/data_flow.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all diagrams."""
    print("Generating AegisAI diagrams...")
    
    create_system_architecture()
    create_anomaly_cascade()
    create_fleet_coordination()
    create_power_flow()
    create_data_flow()
    
    print("\nAll diagrams saved to docs/diagrams/")


if __name__ == "__main__":
    main()
