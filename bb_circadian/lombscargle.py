import bb_behavior.db
from collections import defaultdict
import datetime, pytz
from itertools import chain
import itertools
import matplotlib.pyplot as plt
import numpy as np
import numba
import pandas as pd
import warnings
import scipy.stats
import traceback


def get_shuffled_time_series(datetimes, begin_dt, end_dt, values, unshuffled_values=None, n_iter=400, shuffle_interval_hours=1.0):
    assert shuffle_interval_hours > 0.0
    assert len(values[0]) > 0
    assert datetimes.shape[0] == values[0].shape[0]

    resample_duration = datetime.timedelta(hours=shuffle_interval_hours)
    if unshuffled_values is None:
        unshuffled_values = []

    # Collect the fitting indices once.
    shuffled_idx = []
    current_dt = begin_dt
    while current_dt < end_dt:
        current_end_dt = current_dt + resample_duration
        shuffled_idx.append(np.where((datetimes >= current_dt) & (datetimes < current_end_dt))[0])
        current_dt = current_end_dt

    # And then re-shuffle them.
    for _ in range(n_iter):
        shuffled_idx_idx = range(0, len(shuffled_idx))
        shuffled_idx_idx = np.random.choice(shuffled_idx_idx, replace=True, size=len(shuffled_idx))
        shuffled_idx_flat = [shuffled_idx[i] for i in shuffled_idx_idx]
        shuffled_idx_flat = np.array(list(itertools.chain(*shuffled_idx_flat)), dtype=np.int)

        shuffled_values = [v[shuffled_idx_flat] for v in values]
        min_length = min([v.shape[0] for v in itertools.chain(shuffled_values, unshuffled_values)])
        shuffled_values = [v[:min_length] for v in shuffled_values]
        cut_unshuffled_values = [u[:min_length] for u in unshuffled_values]
        shapes = [v.shape[0] for v in itertools.chain(shuffled_values, cut_unshuffled_values)]
        assert np.all(np.array(shapes) == min_length)
        yield itertools.chain(shuffled_values, cut_unshuffled_values)

@numba.njit
def circadian_sine(x, amplitude, phase, offset):
    frequency = 2.0 * np.pi * 1 / 60 / 60/  24
    return np.sin(x * frequency + phase) * amplitude + offset
@numba.njit
def fixed_minimum_circadian_sine(x, amplitude, phase):
    frequency = 2.0 * np.pi * 1 / 60 / 60/  24
    return np.sin(x * frequency + phase) * amplitude + amplitude

def fit_circadian_sine(X, Y, fix_minimum=False):
    """Fits a sine wave with a circadian frequency to timestamp-value pairs with the timestamps being in second precision.

    Arguments:
        X: np.array
            Timestamps in seconds. Do not have to be sorted.
        Y: np.array
            Values for their respective timestamps.
        fix_minimum: boolean
            Whether to fix offset = amplitude so that min(f(X)) == 0.
    Returns:
        Dictionary with all informations about a fit.
    """
    import scipy.optimize
    
    amplitude = 3 * np.std(Y) / (2 ** 0.5)
    phase = 0
    offset = np.mean(Y)
    if not fix_minimum:
        initial_parameters = [amplitude, phase, offset]
        fun = circadian_sine
    else:
        initial_parameters = [amplitude + offset / 2.0, phase]
        fun = fixed_minimum_circadian_sine
    fit = scipy.optimize.curve_fit(fun, X, Y, p0=initial_parameters)
    circadian_sine_parameters = fit[0]
    y_predicted = fun(X, *circadian_sine_parameters)
    circadian_sse = np.sum((y_predicted - Y) ** 2.0)
    
    from numpy.polynomial.polynomial import Polynomial
    
    constant_fit, full_data = Polynomial.fit(X, Y, deg=0, full=True)
    constant_sse = full_data[0][0]
    
    linear_fit, full_data = Polynomial.fit(X, Y, deg=1, full=True)
    linear_sse = full_data[0][0]
    
    r_squared_linear = 1.0 - (circadian_sse / linear_sse)
    r_squared = 1.0 - (circadian_sse / constant_sse)
    
    return dict(parameters=circadian_sine_parameters, jacobian=fit[1],
                circadian_sse=circadian_sse,
                angular_frequency=2.0 * np.pi * 1 / 60 / 60/  24,
                linear_parameters=linear_fit.convert().coef,
                linear_sse=linear_sse,
                constant_parameters=constant_fit.convert().coef,
                constant_sse=constant_sse,
                r_squared=r_squared, r_squared_linear=r_squared_linear)


def collect_circadianess_data_for_bee_date(bee_id, date, velocities=None,
        delta=datetime.timedelta(days=1, hours=12),
        resample_runs=400, shuffle_interval_hours=1.0, **kwargs):
    
    if velocities is None:
        velocities = bb_behavior.db.trajectory.get_bee_velocities(bee_id, date - delta, date + delta, **kwargs)
    if velocities is None:
        return None
    if "offset" in velocities.columns:
        ts = velocities.offset.values
    else:
        ts = np.array([t.total_seconds() for t in velocities.datetime - velocities.datetime.min()])
    v = velocities.velocity.values
    assert v.shape[0] == ts.shape[0]

    begin_dt = date - delta
    end_dt = date + delta

    # Collect LS powers for shuffled time series.
    resampled_powers = []
    for shuffled_v, shuffled_ts in get_shuffled_time_series(velocities.datetime, begin_dt, end_dt,
                    values=(v,), unshuffled_values=(ts,),
                    shuffle_interval_hours=shuffle_interval_hours, n_iter=resample_runs):
        try:
            fit = fit_circadian_sine(shuffled_ts, shuffled_v)
            resampled_powers.append(fit["r_squared"])
        except (RuntimeError, TypeError):
            continue

    # Distribution of powers.
    dist_args = scipy.stats.chi2.fit(resampled_powers, floc=0.0)
    resampled_distribution = scipy.stats.chi2(*dist_args)
    
    try:
        bee_date_data = fit_circadian_sine(ts, v)
    except (RuntimeError, TypeError):
        return None
    bee_date_data["bee_id"] = bee_id
    bee_date_data["date"] = date
    bee_date_data["r_squared_resampled_mean"] = np.mean(resampled_powers)
    bee_date_data["r_squared_resampled_95"] = np.percentile(resampled_powers, 95)
    bee_date_data["p_value"] = resampled_distribution.sf(bee_date_data["r_squared"])
    bee_date_data["chi2_fit0"], _, bee_date_data["chi2_scale"] = dist_args
    bee_date_data["goodness_of_fit_D"], bee_date_data["goodness_of_fit"] =  scipy.stats.kstest(resampled_powers, resampled_distribution.cdf)

    try:
        fixed_minimum_fit = fit_circadian_sine(ts, v, fix_minimum=True)
        bee_date_data["fixed_minimum_model"] = fixed_minimum_fit
    except (RuntimeError, TypeError):
        pass

    return bee_date_data

def plot_circadian_fit(velocities_df, circadian_fit_data=None):
    import seaborn as sns
    # Convert to timestamps in seconds.
    ts = np.array([t.total_seconds() for t in velocities_df.datetime - velocities_df.datetime.min()])
    
    if circadian_fit_data is None:
        circadian_fit_data = fit_circadian_sine(ts, velocities_df.velocity.values)
    
    velocities_resampled = velocities_df.copy()
    velocities_resampled.set_index("datetime", inplace=True)
    velocities_resampled = velocities_resampled.resample("2min").mean()
    
    ts_resampled = np.array([t.total_seconds() for t in velocities_resampled.index - velocities_df.datetime.min()])
    
    angular_frequency = circadian_fit_data["angular_frequency"]
    amplitude, phase, offset = circadian_fit_data["parameters"]
    b0, b1 = circadian_fit_data["linear_parameters"]
    mean = circadian_fit_data["constant_parameters"][0]

    base_activity = max(0, offset - amplitude)
    max_activity = offset + amplitude
    fraction_circadian = max_activity / (base_activity + max_activity)

    fig, ax = plt.subplots(figsize=(20, 5))
    
    y = np.sin(ts_resampled * angular_frequency + phase) * amplitude + offset
    y_linear = (ts_resampled * b1) + b0

    velocities_resampled["circadian_model"] = y
    velocities_resampled["linear_model"] = y_linear
    velocities_resampled["constant_model"] = mean

    velocities_resampled.plot(y="velocity", ax=ax, color="k", alpha=0.3)
    velocities_resampled.plot(y="circadian_model", ax=ax, color="g", alpha=1.0)
    velocities_resampled.plot(y="linear_model", ax=ax, color="r", linestyle="--", alpha=1.0)
    velocities_resampled.plot(y="constant_model", ax=ax, color="r", linestyle=":", alpha=1.0)

    fixed_minimum_r_squared = None
    fixed_amplitude, fixed_phase = None, None
    if "fixed_minimum_model" in circadian_fit_data:
        fixed_minimum_r_squared = circadian_fit_data["fixed_minimum_model"]["r_squared"]
        fixed_amplitude, fixed_phase = circadian_fit_data["fixed_minimum_model"]["parameters"]
        y = np.sin(ts_resampled * angular_frequency + fixed_phase) * fixed_amplitude + fixed_amplitude
        velocities_resampled["circadian_model_fixed_min"] = y
        velocities_resampled.plot(y="circadian_model_fixed_min", ax=ax, color="b", linestyle=":", alpha=1.0)

    ax.axhline(base_activity, color="k", linestyle="--")
    ax.axhline(max_activity, color="k", linestyle="--")
    plt.ylim(0, offset + amplitude * 5.0)
    plt.title("R^2 (vs constant): {:3.3f}, R^2 (vs linear): {:3.3f}\n" \
                "circadian: {:2.1%}, amplitude: {:3.3f}, phase: {:3.1f}h\n" \
                "R^2 of zero-min model (vs constant): {:3.3f}, amplitude: {:3.3f}, phase: {:3.1f}h".format(
        circadian_fit_data["r_squared"],
        circadian_fit_data["r_squared_linear"],
        fraction_circadian,
        2.0 * amplitude, phase,
        fixed_minimum_r_squared, 2.0 * fixed_amplitude, fixed_phase))
    plt.show()

def collect_circadianess_subsamples_for_bee_date(bee_id, date, verbose=False, **kwargs):
    delta = datetime.timedelta(days=1, hours=12)
    velocities = bb_behavior.db.trajectory.get_bee_velocities(bee_id, date - delta, date + delta, **kwargs)
    if velocities is None:
        return []
    starting_date = date - delta
    ending_date = date + delta
    total_seconds = (ending_date - starting_date).total_seconds()
    assert starting_date.tzinfo == pytz.UTC
    assert velocities.datetime.iloc[0].tzinfo == pytz.UTC
    assert starting_date <= velocities.datetime.min()

    velocities["offset"] = [t.total_seconds() for t in velocities.datetime - starting_date]
    
    interval_duration_seconds = 60 * 5
    
    all_results = dict()
    for subsample_hours in (None, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 12.0, 18.0):
        subsampled_velocities = velocities
        if subsample_hours is not None:
            valid_indices = np.zeros(shape=(velocities.shape[0],), dtype=np.bool)
            intervals = np.arange(0.0, total_seconds, subsample_hours * 3600)
            for begin in intervals:
                end = begin + interval_duration_seconds
                valid_indices[(velocities.offset.values >= begin) & (velocities.offset.values < end)] = True
            subsampled_velocities = velocities.iloc[valid_indices, :]
        try:
            results = collect_circadianess_data_for_bee_date(bee_id, date, velocities=subsampled_velocities, **kwargs)
        except Exception as e:
            results = dict(error=str(e), stacktrace=traceback.format_exc())
        if results is not None:
            results["subsampling"] = subsample_hours
            all_results[subsample_hours] = results
    return all_results

def get_circadianess_per_age_groups(date, bees_per_group=None, max_workers=32, verbose=None, progress=None):
    assert date.tzinfo == pytz.UTC

    from concurrent.futures import ProcessPoolExecutor
    from .meta import get_bee_age_groups
    
    age_groups = get_bee_age_groups(date)
    all_results = dict()
    
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
            results = []
            for bee in bees:
                bee = int(bee)
                results.append(executor.submit(collect_circadianess_subsamples_for_bee_date,
                                              bee, date))
            # Collect results.
            for future, bee in zip(results, bees): 
                result = future.result()
                if not result:
                    continue
                
                for subsample, sub_dict in result.items():
                    if sub_dict is not None:
                        sub_dict["group"] = group
                        all_results[subsample, date, bee] = sub_dict
                    
    return all_results

def get_ls_power_for_bee_date(bee_id, date, velocities = None, verbose=False,
                              resample_interval_hours=1.0, resample_runs=400, **kwargs):
    import seaborn as sns
    from astropy.stats import LombScargle
    verbose = verbose or 0
    delta = datetime.timedelta(days=1, hours=12)
    if velocities is None:
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

            # Collect LS powers for shuffled time series.
            resampled_powers = []
            for shuffled_v, shuffled_ts in get_shuffled_time_series(velocities.datetime, date - delta, date + delta,
                            values=(v,), unshuffled_values=(ts,),
                            shuffle_interval_hours=resample_interval_hours, n_iter=resample_runs):
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
            raise
        
    return circadian_power, circadian_amplitude,\
            goodness_of_distribution_D, goodness_of_distribution, \
            p_value, resampled_powers

def get_ls_power_subsamples_for_bee_date(bee_id, date, verbose=False, **kwargs):
    delta = datetime.timedelta(days=1, hours=12)
    velocities = bb_behavior.db.trajectory.get_bee_velocities(bee_id, date - delta, date + delta, **kwargs)
    if velocities is None:
        return []
    starting_date = date - delta
    #print(starting_date)
    ending_date = date + delta
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