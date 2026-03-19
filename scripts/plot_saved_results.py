import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


def plot_results(result_dir: str = "result", basename: str = "gw_prefilter_per_eval") -> None:
    xlsx_path = os.path.join(result_dir, f"{basename}.xlsx")
    csv_path = os.path.join(result_dir, f"{basename}.csv")

    if os.path.exists(xlsx_path):
        df_eval = pd.read_excel(xlsx_path)
    elif os.path.exists(csv_path):
        df_eval = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No saved results found: {xlsx_path} or {csv_path}")

    long_rows = []
    for _, r in df_eval.iterrows():
        long_rows.append({'model': r['model'], 'patient': r['patient'], 'seizure': r.get('seizure_no', None),
                          'metric': 'Acc@N',  'condition': 'With', 'value': r['with_acc_all']})
        long_rows.append({'model': r['model'], 'patient': r['patient'], 'seizure': r.get('seizure_no', None),
                          'metric': 'Acc@N',  'condition': 'No',   'value': r['no_acc_all']})
        long_rows.append({'model': r['model'], 'patient': r['patient'], 'seizure': r.get('seizure_no', None),
                          'metric': 'Hit@20', 'condition': 'With', 'value': r['with_hit20']})
        long_rows.append({'model': r['model'], 'patient': r['patient'], 'seizure': r.get('seizure_no', None),
                          'metric': 'Hit@20', 'condition': 'No',   'value': r['no_hit20']})

    plot_df = pd.DataFrame(long_rows)
    plot_df = plot_df.dropna(subset=['value'])
    plot_df = plot_df[plot_df['value'] > 0]

    sns.set_theme(style='whitegrid', context='talk')

    for metric in ['Acc@N', 'Hit@20']:
        sub = plot_df[plot_df['metric'] == metric].copy()
        if sub.empty:
            continue
        plt.figure(figsize=(8, 6))
        ax = sns.boxplot(data=sub, x='model', y='value', hue='condition', width=0.6)
        # keep box legend
        handles_box, labels_box = ax.get_legend_handles_labels()
        # dots without legend
        sns.stripplot(data=sub, x='model', y='value', hue='condition', dodge=True, color='k', size=4, alpha=0.6, ax=ax, legend=False)
        ax.legend(handles_box, labels_box, title='Condition')

        # significance (paired by patient+seizure)
        for i, m in enumerate(sub['model'].unique()):
            sm = sub[sub['model'] == m]
            pvt = sm.pivot_table(index=['patient','seizure'], columns='condition', values='value', aggfunc='first').dropna()
            if pvt.empty or 'With' not in pvt.columns or 'No' not in pvt.columns:
                continue
            try:
                stat, p = wilcoxon(pvt['With'], pvt['No'])
            except ValueError:
                continue
            stars = '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'ns'
            ymax = sm['value'].max()
            y = ymax + 0.05 * (sub['value'].max() - sub['value'].min() + 1e-6)
            ax.text(i, y, stars, ha='center', va='bottom', color='crimson', fontsize=12, fontweight='bold')

        ax.set_ylabel(metric)
        ax.set_xlabel('Model')
        plt.tight_layout()
        plt.show()
