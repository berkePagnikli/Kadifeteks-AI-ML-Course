import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CancerDataAnalysis:
    def __init__(self, data_path):
        """Initialize the data analysis class"""

        self.data_path = data_path
        self.df = None
        self.target_col = None
        
    def load_data(self):
        """Load the dataset and perform initial exploration"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        
        # Identify target column (diagnosis column)
        self.target_col = self.df.columns[0]  # First column is diagnosis
        
        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"Target column: {self.target_col}")
        print("\nDataset Info:")
        print(self.df.info())
        
        return self.df
    
    def basic_statistics(self):
        """Display basic statistics about the dataset"""
        print("\n" + "="*50)
        print("BASIC STATISTICS")
        print("="*50)
        
        print("\nDescriptive Statistics:")
        print(self.df.describe())
        
        print(f"\nMissing values:")
        missing_vals = self.df.isnull().sum()
        if missing_vals.sum() == 0:
            print("No missing values found!")
        else:
            print(missing_vals[missing_vals > 0])
    
    def class_imbalance_analysis(self, save_path="Graphs/class_distribution.png"):
        """Analyze and visualize class imbalance"""

        
        # Calculate class distribution
        class_counts = self.df[self.target_col].value_counts()
        class_percentages = self.df[self.target_col].value_counts(normalize=True) * 100
        
        print(f"\nClass Distribution:")
        for class_val, count in class_counts.items():
            class_name = "Malignant" if class_val == 1 else "Benign"
            percentage = class_percentages[class_val]
            print(f"{class_name} (Class {class_val}): {count} samples ({percentage:.2f}%)")
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        class_labels = ["Benign (0)", "Malignant (1)"]
        colors = ['lightblue', 'lightcoral']
        
        bars = ax1.bar(class_labels, class_counts.values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Samples', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        ax2.pie(class_counts.values, labels=class_labels, colors=colors, autopct='%1.1f%%', 
                startangle=90, explode=(0.05, 0.05))
        ax2.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    
    def correlation_analysis(self, save_path="Graphs/correlation_heatmap.png"):
        """Perform correlation analysis and create heatmap"""

        print("\n" + "="*50)
        print("CORRELATION ANALYSIS")
        print("="*50)
        
        # Calculate correlation matrix
        correlation_matrix = self.df.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(20, 16))
        
        # Generate heatmap
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": .8},
                   annot_kws={'size': 8})
        
        plt.title('Feature Correlation Heatmap', fontsize=18, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self):
        """Run the complete data analysis pipeline"""

        print("🔍 Starting Complete Cancer Dataset Analysis")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Basic statistics
        self.basic_statistics()
        
        # Class imbalance analysis
        self.class_imbalance_analysis()
        
        # Correlation analysis
        self.correlation_analysis()
        
        print("\n" + "="*60)
        print("✅ Analysis Complete! All visualizations have been saved.")
        print("="*60)
        
        # Summary
        print("\nANALYSIS SUMMARY:")
        print(f"   • Dataset shape: {self.df.shape}")
        print(f"   • Number of features: {len(self.df.columns) - 1}")
        print(f"   • Target variable: {self.target_col}")
        print("   • Generated visualizations:")
        print("     - class_distribution.png")
        print("     - correlation_heatmap.png") 
        print("     - feature_distributions.png")


def main():
    """Main function to run the analysis"""
    # Initialize analysis
    data_path = "cancer.csv"
    analysis = CancerDataAnalysis(data_path)
    
    # Run complete analysis
    analysis.run_complete_analysis()


if __name__ == "__main__":
    main()