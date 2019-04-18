from astropy.stats import LombScargle
import bb_behavior.db
from collections import defaultdict
import datetime, pytz
from itertools import chain
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import scipy.stats

def get_ls_power_for_bee_date(bee_id, date, velocities = None, verbose=False,
                              resample_interval_hours=1.0, resample_runs=200, **kwargs):
    verbose = verbose or 0
    if velocities is None:
        delta = datetime.timedelta(days=1)
        velocities = bb_behavior.db.trajectory.get_bee_velocities(bee_id, date - delta, date + delta, **kwargs)
    if velocities is None:
        return None, None
    if "offset" in velocities.columns:
        ts = velocities.offset.values
    else:
        ts = np.array([t.total_seconds() for t in velocities.datetime - velocities.datetime.min()])
    v = velocities.velocity.values
    assert v.shape[0] == ts.shape[0]
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # First, shuffle all the data.
            resample_duration = datetime.timedelta(hours=resample_interval_hours)
            resampled_powers = []
            
            # Collect the fitting indices once.
            shuffled_idx = []
            begin_dt = date - datetime.timedelta(days=1)
            end_dt = date + datetime.timedelta(days=1)
            current_dt = begin_dt
            while current_dt < end_dt:
                current_end_dt = current_dt + resample_duration
                shuffled_idx.append(np.where((velocities.datetime >= current_dt) & (velocities.datetime < current_end_dt))[0])
                current_dt = current_end_dt
            # And then re-shuffle them.
            for _ in range(resample_runs):
                #np.random.shuffle(shuffled_idx)
                shuffled_idx_idx = range(0, len(shuffled_idx))
                shuffled_idx_idx = np.random.choice(shuffled_idx_idx, replace=True, size=len(shuffled_idx))
                #assert len(shuffled_idx_idx) == len(shuffled_idx)
                shuffled_idx_flat = [shuffled_idx[i] for i in shuffled_idx_idx]
                #print(len(shuffled_idx_flat), velocities.shape[0])
                shuffled_idx_flat = np.array(list(itertools.chain(*shuffled_idx_flat)), dtype=np.int)
                #print(len(shuffled_idx_flat))
                #assert shuffled_idx_flat.shape[0] == velocities.shape[0]
                #assert shuffled_idx_flat.shape[0] == ts.shape[0]
                
                shuffled_v = v[shuffled_idx_flat]
                l = min(ts.shape[0], shuffled_v.shape[0])
                shuffled_ts = ts[:l]
                shuffled_v = shuffled_v[:l]
                
                """np.random.shuffle(shuffled_idx)
                shuffled_idx_flat = np.array(list(itertools.chain(*shuffled_idx)), dtype=np.int)
                shuffled_v = v[shuffled_idx_flat]
                shuffled_ts = ts"""
                
                #print(shuffled_v)
                #print(shuffled_ts)
                assert shuffled_v.shape[0] == shuffled_ts.shape[0]
                ls = LombScargle(shuffled_ts, shuffled_v)
                circadian_power = ls.power((1 / 60 / 60 / 24))
                resampled_powers.append(circadian_power)
            # Distribution of powers.
            dist_args = scipy.stats.chi2.fit(resampled_powers, floc=0.0)
            resampled_distribution = scipy.stats.chi2(*dist_args)
            #resampled_distribution = scipy.stats.halfnorm(loc=0, scale=np.std(resampled_powers))
            goodness_of_distribution_D, goodness_of_distribution = scipy.stats.kstest(resampled_powers, resampled_distribution.cdf)
            
            
            # Now do it for the one sample.
            ls = LombScargle(ts, v)
            
            circadian_power = ls.power((1 / 60 / 60 / 24))
            circadian_fit = ls.model(ts, (1 / 60 / 60 / 24))
            circadian_amplitude = np.max(circadian_fit) - np.min(circadian_fit)
            #D, p_value = scipy.stats.ks_2samp(resampled_powers, [circadian_power])
            p_value = resampled_distribution.sf(circadian_power)
            
            if verbose >= 1:
                fig, ax = plt.subplots(figsize=(20, 5))
                sns.distplot(np.log1p(resampled_powers), kde=False)
                #sns.distplot(np.log1p(resampled_powers))
                rv = scipy.stats.halfnorm(loc=0, scale=np.std(resampled_powers))
                x = np.linspace(0, np.max(resampled_powers))
                ax = ax.twinx()
                ax.plot(x, rv.pdf(x), 'k-', lw=1)
                #sns.distplot(r, hist=False)
                plt.show()
            if verbose >= 2:
                frequency, power = ls.autopower(maximum_frequency=(1 / 60 / 60 / 24) * 10, 
                                        samples_per_peak=25)#, method="fastchi2")
                fig, ax = plt.subplots(figsize=(20, 5))
                ax.plot(frequency * 60 * 60 * 24, power)
                ax.set_ylim(0, 1)
                for level in ls.false_alarm_level([0.1, 0.05, 0.01]):
                    ax.axhline(level, linestyle=":", color="k")
                ax.axhline(np.mean(resampled_powers), linestyle="--", color="r")
                ax.text(0, 0, "{:3.2f}".format(p_value))
                plt.show()
    except Exception as e:
        circadian_power, circadian_amplitude, goodness_of_distribution_D, goodness_of_distribution, p_value, resampled_powers = \
                None, None, None, None, None, None
        error_string = str(e)
        if error_string != "Singular matrix":
            print(error_string)
            #raise
        
    return circadian_power, circadian_amplitude,\
            goodness_of_distribution_D, goodness_of_distribution, \
            p_value, resampled_powers

def get_ls_power_subsamples_for_bee_date(bee_id, date, verbose=False, **kwargs):
    delta = datetime.timedelta(days=1)
    velocities = bb_behavior.db.trajectory.get_bee_velocities(bee_id, date - delta, date + delta, **kwargs)
    if velocities is None:
        return []
    starting_date = date - datetime.timedelta(days=1)
    #print(starting_date)
    ending_date = date + datetime.timedelta(days=1)
    total_seconds = (ending_date - starting_date).total_seconds()
    assert starting_date.tzinfo == pytz.UTC
    assert velocities.datetime.iloc[0].tzinfo == pytz.UTC
    assert starting_date <= velocities.datetime.min()

    #print(velocities.datetime.values[0])
    velocities["offset"] = [t.total_seconds() for t in velocities.datetime - starting_date]
    
    interval_duration_seconds = 60 * 5
    all_resampled_powers = []
    
    results_dataframe = list()
    for subsample_hours in (None, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 12.0, 18.0):
        subsampled_velocities = velocities
        if subsample_hours is not None:
            valid_indices = np.zeros(shape=(velocities.shape[0],), dtype=np.bool)
            intervals = np.arange(0.0, total_seconds, subsample_hours * 3600)
            for begin in intervals:
                end = begin + interval_duration_seconds
                valid_indices[(velocities.offset.values >= begin) & (velocities.offset.values < end)] = True
            subsampled_velocities = velocities.iloc[valid_indices, :]
        (power, amplitude, \
            goodness_of_distribution_D, goodness_of_distribution, \
            p_value, powers) = get_ls_power_for_bee_date(bee_id, date,
                                                     velocities=subsampled_velocities,
                                                     verbose=verbose, **kwargs)
        results_dataframe.append(dict(bee_id=bee_id, date=date,
                                      subsample=subsample_hours, power=power, amplitude=amplitude,
                                      goodness_of_distribution_D=goodness_of_distribution_D,
                                      goodness_of_distribution=goodness_of_distribution, p_value=p_value,
                                      n_data_points=velocities.shape[0],
                                      n_subsampled_data_points=subsampled_velocities.shape[0]))
        all_resampled_powers.append(powers)
    return results_dataframe, all_resampled_powers
        
def get_ls_powers_per_age_groups(date, bees_per_group=None, max_workers=32, verbose=None, progress=None):
    assert date.tzinfo == pytz.UTC

    from concurrent.futures import ProcessPoolExecutor
    from .meta import get_bee_age_groups
    
    age_groups = get_bee_age_groups(date)
    all_dataframes = []
    all_resampled_powers = []
    
    progress_bar = lambda x: x
    if progress == "tqdm":
        import tqdm
        progress_bar = tqdm.tqdm
    elif progress == "tqdm_notebook":
        import tqdm
        progress_bar = tqdm.tqdm_notebook

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for group, bees in progress_bar(list(sorted(age_groups.items()))):
            if bees_per_group is not None and len(bees) > bees_per_group:
                bees = np.random.choice(list(bees), replace=False, size=bees_per_group)
            
            # Query execution.
            dataframes = []
            for bee in bees:
                bee = int(bee)
                dataframes.append(executor.submit(get_ls_power_subsamples_for_bee_date,
                                              bee, date, progress=None, verbose=verbose))
            # Collect results.
            for idx, future in enumerate(dataframes): 
                result = future.result()
                if not result:
                    continue
                dfs, resampled_powers = result
                for df in dfs:
                    if df is not None:
                        df["group"] = group
                        all_dataframes.append(df)
                #for powers in resampled_powers:
                #    all_resampled_powers.append(powers)
            #print("Age {} to {}\tMedian power: {:4.2f}".format(*group, np.median(powers)))
    all_dataframes = pd.DataFrame(all_dataframes)
    return all_dataframes#, all_resampled_powers