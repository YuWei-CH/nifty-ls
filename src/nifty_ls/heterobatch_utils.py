import numpy as np

def validate_frequency_grid_mp(
    fmin, fmax, Nf, t_list, assume_sorted_t=True, samples_per_peak=5, nyquist_factor=5
):
    if t_list is None:
        raise ValueError("t_list must be provided as a list of time arrays.")
    N_series = len(t_list)

    # Helper to broadcast scalars to lists
    def _broadcast(x, name, cast_fn):
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) != N_series:
                raise ValueError(f"Length of '{name}' must match number of series ({N_series}).")
            return list(x)
        else:
            return [x] * N_series
    
    fmin_vals = _broadcast(fmin, "fmin", float)
    fmax_vals = _broadcast(fmax, "fmax", float)
    Nf_vals   = _broadcast(Nf,   "Nf",   int)

    fmin_lst = []
    df_lst = []
    Nf_lst = []

    baseline_lst = []
    for ti in t_list:
        if ti.size < 2:
            raise ValueError("Each time array must have at least two points.")
        b = (ti[-1] - ti[0]) if assume_sorted_t else np.ptp(ti)
        if b <= 0:
            raise ValueError("Time array must be non-degenerate and sorted if assume_sorted_t=True.")
        baseline_lst.append(float(b))
    
    for i, ti in enumerate(t_list):
        baseline = baseline_lst[i]
        target_df = 1.0 / (samples_per_peak * baseline)

        if fmax_vals[i] is None:
            avg_nyquist = 0.5 * ti.size / baseline
            fmax_i = avg_nyquist * nyquist_factor
        else:
            fmax_i = float(fmax_vals[i])

        if fmin_vals[i] is None:
            fmin_i = target_df / 2.0
        else:
            fmin_i = float(fmin_vals[i])

        if Nf_vals[i] is None:
            Nf_i = 1 + int(np.round((fmax_i - fmin_i) / target_df))
        else:
            Nf_i = int(Nf_vals[i])

        if fmin_i >= fmax_i:
            raise ValueError(f"fmin({fmin_i}) ≥ fmax({fmax_i}) at index {i}.")
        
        if Nf_i < 1:
            raise ValueError(f"Nf at index {i} must be positive, got {Nf_i}.")
        
        df_i = (fmax_i - fmin_i) / (Nf_i - 1)
        if df_i <= 0:
            raise ValueError(f"Computed df({df_i}) must be positive at index {i}.")
        
        fmin_lst.append(fmin_i)
        df_lst.append(df_i)
        Nf_lst.append(Nf_i)

    return fmin_lst, df_lst, Nf_lst

def gen_data_mp(N_series=100_000, N_batch=None, N_d=100, dtype=np.float64, seed=5043):
    rng = np.random.default_rng(seed)

    # allow lengths from 50% to 150% of N_d
    min_len = max(1, int(N_d * 0.5))
    max_len = int(N_d * 1.5)

    t_list = []
    y_list = []
    dy_list = []

    N_batch = N_batch if N_batch else 1

    for _ in range(N_series):
        # random series length
        N_d_i = rng.integers(min_len, max_len + 1)
        t_i = np.sort(rng.random(N_d_i, dtype=dtype)) * 123

        if N_batch:
            freqs = rng.random((N_batch, 1), dtype=dtype) * 10 + 1
            # broadcast over time: (N_batch, N_d_i)
            y_i = np.sin(freqs * t_i) + 1.23
            dy_i = rng.random((N_batch, N_d_i), dtype=dtype) * 0.1 + 0.01
            noise = rng.normal(0, dy_i, size=(N_batch, N_d_i))
            y_i = y_i + noise

        # make read-only
        t_i.setflags(write=False)
        y_i.setflags(write=False)
        dy_i.setflags(write=False)
        
        t_list.append(t_i)
        y_list.append(y_i)
        dy_list.append(dy_i)

    fmin_lst, df_lst, Nf_lst = validate_frequency_grid_mp(
        fmin=None, fmax=None, Nf=None, t_list=t_list)

    fmax_lst = [fmin_lst[i] + df_lst[i] * (Nf_lst[i] - 1) for i in range(len(fmin_lst))]

    return dict(t=t_list, y=y_list, dy=dy_list, fmin=fmin_lst, fmax=fmax_lst, df=df_lst, Nf=Nf_lst)