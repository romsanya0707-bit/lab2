"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

import pandas as pd

def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:

    df = pd.read_csv(data.csv_path, sep='\t', na_values='NA')


    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']


    male_df = df[df['Gender'] == 'Male']
    female_df = df[df['Gender'] == 'Female']


    men_count = int(len(male_df))
    women_count = int(len(female_df))


    men_mri_correlation = {}
    women_mri_correlation = {}


    max_abs_corr = -1.0
    strongest_mri_feature = ""


    for feature in features:

        m_corr = male_df[feature].corr(male_df['MRI_Count'], method='pearson')
        men_mri_correlation[feature] = float(m_corr)


        f_corr = female_df[feature].corr(female_df['MRI_Count'], method='pearson')
        women_mri_correlation[feature] = float(f_corr)


        if abs(m_corr) > max_abs_corr:
            max_abs_corr = abs(m_corr)
            strongest_mri_feature = feature

        if abs(f_corr) > max_abs_corr:
            max_abs_corr = abs(f_corr)
            strongest_mri_feature = feature


    return BrainCorrelationSummary(
        men_count=men_count,
        women_count=women_count,
        men_mri_correlation=men_mri_correlation,
        women_mri_correlation=women_mri_correlation,
        strongest_mri_feature=strongest_mri_feature
)
 


