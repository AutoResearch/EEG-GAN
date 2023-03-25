---
hide:
    -toc
---

# Data Structure

We use a very simple 2-dimensional data structure where each row is a different trial-level sample that encompasses both the condition in the first column and then the EEG time series:

| Condition | Time1 | Time2 | Time3 | Time4 | ... |
| --- | --- | --- | --- | --- | --- |
| 0 | 2.28 | -4.44 | -0.98 | -5.67 | ... |
| 1 | 11.67 | 0.66 | 1.43 | 11.62 | ... |
| 1 | 11.90 | 8.67 | 1.85 | 0.73 | ... |
| 0 | 6.73 | 3.63 | 3.80 | 5.63 | ... |
| ... | ... | ... | ... | ... | ... |

Notes:
<ul>
<li> The package is currently designed to use one electrode of data per sample; however, we are actively developping its use with full electrode densities. </li>
<li> The package does not consider individual differences of participant and so we do not provide it any participant IDs. </li>
<li> The time series can be as long as you like with the understanding that computational time is proportional to the length of the time series. </li>
<li> Data must be saved as csv files.
</ul>



