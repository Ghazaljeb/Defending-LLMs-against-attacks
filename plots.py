import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the correct path
file_path = "E:/فصل 6/Final_Result_Experiment.csv"

try:
    # Load the data
    df = pd.read_csv(file_path)

    # Set aesthetic style
    sns.set_theme(style="whitegrid")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # --- 1. Pie Chart: Category Distribution (Adverserial vs Benign) ---
    ax1 = plt.subplot(2, 2, 1)
    category_counts = df['category'].value_counts()
    colors = sns.color_palette('pastel')[0:2]
    ax1.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140, colors=colors, explode=(0.05, 0))
    ax1.set_title('Distribution of Input Categories', fontsize=14, fontweight='bold')

    # --- 2. Bar Chart: Guard Decision by Subtype (For Adversarial) ---
    # Filter only adversarial samples to see how Guard handled different attacks
    adversarial_df = df[df['category'] == 'Adverserial']
    ax2 = plt.subplot(2, 2, 2)
    
    # We want to see the count of guard_decision per subtype within Adverserial category
    # Specifically looking for how many were BLOCKED_BY_GUARDRAIL
    subtype_guard_counts = adversarial_df.groupby(['subtype', 'guard_decision']).size().unstack(fill_value=0)
    
    if not subtype_guard_counts.empty:
        subtype_guard_counts.plot(kind='bar', stacked=True, ax=ax2, color=['#ff9999','#66b3ff'])
        ax2.set_title('Guard Decisions per Attack Subtype', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Count')
        ax2.set_xlabel('Attack Subtype')
        plt.setp(ax2.get_xticklabels(), rotation=45)
    else:
        ax2.text(0.5, 0.5, 'No Adversarial Data Found', ha='center')

    # --- 3. Bar Chart: Status Distribution (Overall Performance) ---
    ax3 = plt.subplot(2, 2, 3)
    status_counts = df['status'].value_counts()
    sns.barplot(x=status_counts.index, y=status_counts.values, ax=ax3, palette='viridis')
    ax3.set_title('Overall Experiment Status', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Count')
    for i, v in enumerate(status_counts.values):
        ax3.text(i, v + 0.1, str(v), ha='center', fontweight='bold')

    # --- 4. Success Rate in Adversarial (Percentage) ---
    # Calculate how many adversarial attacks were actually blocked
    ax4 = plt.subplot(2, 2, 4)
    total_adversarial = len(adversarial_df)
    if total_adversarial > 0:
        blocked_adversarial = len(adversarial_df[adversarial_df['guard_decision'] == '[BLOCKED_BY_GUARDRAIL]'])
        success_rate = (blocked_adversarial / total_adversarial) * 100
        
        # Plotting as a simple single bar for emphasis
        ax4.bar(['Attack Block Rate'], [success_rate], color='#4CAF50', width=0.5)
        ax4.set_ylim(0, 110)
        ax4.set_title(f'Adversarial Robustness: {success_rate:.1f}% Blocked', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Percentage (%)')
        ax4.text(0, success_rate + 2, f'{success_rate:.1f}%', ha='center', fontsize=14, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No Adversarial Data', ha='center')

    plt.tight_layout()
    plt.savefig("E:/فصل 6/experiment_analysis_plots.png", dpi=300)
    print("Plots saved successfully to E:/فصل 6/experiment_analysis_plots.png")
    
    # Also print some summary statistics for the user
    print("\n--- Summary Statistics ---")
    print(f"Total Samples: {len(df)}")
    print(f"Adversarial Samples: {len(adversarial_df)}")
    print(f"Benign Samples: {len(df[df['category'] == 'Benign'])}")
    if total_adversarial > 0:
        print(f"Attack Success Rate (Blocked/Total Adversarial): {(blocked_adversarial/total_adversarial)*100:.2f}%")

except Exception as e:
    print(f"An error occurred: {e}")
