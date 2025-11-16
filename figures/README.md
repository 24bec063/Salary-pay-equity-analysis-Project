# Figures Directory

This directory contains visualizations and plots generated from the Machine Learning analysis.

## Generated Figures

The notebooks generate the following visualizations (currently displayed in Jupyter):

### Salary Prediction Analysis
- Salary distribution histogram
- Salary vs Years of Experience scatter plot
- Top 10 countries by median salary
- Median salary by education level
- Top 10 developer roles by median salary
- Correlation heatmap of numerical features

### Model Performance
- Actual vs Predicted salary scatter plot (Random Forest)
- Residual plots
- Residual distribution histogram

### Feature Analysis
- Feature importance bar chart (Top 20 features from Random Forest)

## Saving Figures

To save figures as images, modify the notebook cells:
```python
plt.savefig('figures/figure_name.png', dpi=300, bbox_inches='tight')
```
