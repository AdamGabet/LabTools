import numpy as np
import pandas as pd
from LabData.DataLoaders.SubjectLoader import SubjectLoader
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
import os


def date_to_research_stage(reg_ids, dates):
    """
    Convert a list of dates to research stages based on subject and body measures data.

    Parameters:
    dates: any format of dates, including 'YYYY-MM-DD', 'MM/DD/YYYY', etc and datetime objects.
    reg_ids (list of int): List of registration IDs corresponding to subjects.

    Returns:
    list of str: List of research stages corresponding to the input dates.
    """
    # Load subject and body measures data
    subjects_loader = SubjectLoader()
    subject_data = subjects_loader.get_data(reg_ids=reg_ids).df
    loader = BodyMeasuresLoader()
    body_loader = loader.get_data(reg_ids=reg_ids)
    date_research = get_date_and_research(body_loader)

    # date researcghh has columns RegistrationCode, Date, research_stage
    # find research stage for each reg_id and date the logic is the research stage on closest date on or after the given date
    research_stages = []
    valid_ids = []
    for reg_id, date in zip(reg_ids, dates):
        date = pd.to_datetime(date)
        # Remove timezone info if present to ensure compatibility
        if hasattr(date, 'tz') and date.tz is not None:
            date = date.tz_localize(None)

        subset = date_research[date_research['RegistrationCode'] == reg_id].copy()

        # Ensure subset Date column is also tz-naive
        if subset['Date'].dt.tz is not None:
            subset['Date'] = subset['Date'].dt.tz_localize(None)

        valid_dates = subset[subset['Date'] >= date]
        if not valid_dates.empty:
            closest_date = valid_dates.loc[valid_dates['Date'].idxmin()]
            research_stages.append(closest_date['research_stage'])
            valid_ids.append(reg_id)

        else:
            after_dates = subset[subset['Date'] < date]
            if not after_dates.empty:
                closest_date = after_dates.loc[after_dates['Date'].idxmax()]
                research_stages.append(closest_date['research_stage'])
                valid_ids.append(reg_id)

            else:
                print(f"{subset}")
    return research_stages, valid_ids


def get_date_and_research(loader):
    if 'RegistrationCode' in loader.df_metadata.columns:
        loader.df = loader.df.join(loader.df_metadata[['RegistrationCode', 'research_stage']],
                                   how='inner').reset_index()
    else:
        loader.df = loader.df.join(loader.df_metadata[['research_stage']], how='inner').set_index(['research_stage'],
                                                                                                  append=True).reset_index()
    return loader.df[["RegistrationCode", 'Date', 'research_stage']]

def gait_id_date2research_stage(subject_id, date, research_stage_df=None):
    """
    Map subject ID and recording date to research stage.
    """
    if research_stage_df is None and os.path.exists("/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/skeleton_data/id_date_long.csv"):
        research_stage_df = pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/skeleton_data/id_date_long.csv")

    record = research_stage_df[
        (research_stage_df['subject_id'] == subject_id) &
        (research_stage_df['date'] == date)
    ]
    if not record.empty:
        return record.iloc[0]['research_stage']

    return None

def gait_ids_dates2research_stages(subject_ids, dates, research_stage_df=None):
    """
    Map lists of subject IDs and recording dates to research stages.
    """
    if research_stage_df is None and os.path.exists("/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/skeleton_data/id_date_long.csv"):
        research_stage_df = pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/skeleton_data/id_date_long.csv")

    research_stages = []
    for subject_id, date in zip(subject_ids, dates):
        stage = gait_id_date2research_stage(subject_id, date, research_stage_df)
        research_stages.append(stage)

    return research_stages

if __name__ == "__main__":
    # Example usage
    reg_ids = ["10K_1008294272"]
    dates = ['2022-01-15']
    stages = date_to_research_stage(reg_ids, dates)
    print(stages)