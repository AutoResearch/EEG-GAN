# Data Structure

We use a very simple 2-dimensional data structure where each row is a different trial-level sample that encompasses metadata (e.g., condition, electrode) and then the EEG time series:

| Condition | Electrode | Time1 | Time2 | Time3 | Time4 | ... |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 2.28 | -4.44 | -0.98 | -5.67 | ... |
| 1 | 2 | 11.67 | 0.66 | 1.43 | 11.62 | ... |
| 1 | 1 | 11.90 | 8.67 | 1.85 | 0.73 | ... |
| 0 | 2 | 6.73 | 3.63 | 3.80 | 5.63 | ... |
| ... | ... | ... | ... | ... | ... | ... |

## Data Parameters

The autoencoder and GAN functions require parameters to be set for the data structure. The following parameters inform the model which columns correspond to the condition(s), channels, and time series data. Other columns in the data file are ignored. Channel (`kw_channel`) and Condition (`kw_condition`) parameters are optional but the Time (`kw_time`) parameter is required and defaulted to `Time`.

`kw_time = 'Time'` <br>
`kw_channel = 'Electrode'` <br>
`kw_condition = 'Condition'` <br>

The `kw_time` value corresponds to the substring which all time step columns have in common. This substring is required to be used only by the time step columns. The `kw_condition` can be given as a comma-separated list of multiple conditions e.g. `kw_condition=Condition1,Condition2`.

Notes:
<ul>
<li> The package does not consider individual differences of participant and so we do not provide it any participant IDs. </li>
<li> The time series can be as long as you like with the understanding that computational time is proportional to the length of the time series. </li>
<li> Data must be saved as csv files.
</ul>



