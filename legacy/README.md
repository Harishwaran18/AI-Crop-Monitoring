# Legacy prototypes

These are the original single-file demos from the first iteration. They are kept
for reference and are **not** part of the supported platform.

| Legacy file | Replaced by |
| --- | --- |
| `plant_demo.py` | `cropmon.data.synthetic` + `cropmon.cli demo-data` |
| `ai_crop_monitoring_multispectral.py` | `dashboard/app.py` (Spectral health tab) |
| `crop_monitoring_prototype.py` | `cropmon.models.deep` + `cropmon.pipeline` |
| `smsandwifi.py`, `sms_streamlit.py`, `phonenumber.py` | `cropmon.alerts` (rules + channels) |
| `tomato_disease_app.py` | `cropmon.models.fusion` + dashboard Model tab |
| MATLAB index montage | `matlab/index_panel.m` + `cropmon.viz.index_panel` |
| `matlab_output/CropHealthSummary.csv` | `data/class_summary.csv` (written by the pipeline) |

Changes made while archiving:

* fixed the `SyntaxError` in `sms_streamlit.py` (the recipient is interpolated now)
* removed committed phone numbers and Twilio placeholders; everything reads from
  environment variables (`ALERT_RECIPIENT`, `TWILIO_*`, `FAST2SMS_API_KEY`)
* replaced the hardcoded `C:\Users\...` path in `plant_demo.py`

Known limitations that motivated the rewrite: untrained models presented as
predictions, random "multispectral" channels that were discarded before use, no
caching, and duplicated label sets across apps.

## Archived MATLAB output

`matlab_output/CropHealthSummary.csv` is the original classification summary:
**21024 of 21025 pixels labelled `PestRisk`, one `Stressed`, zero `Healthy`**.
That is not a field result, it is a bug: the if/elseif chain fell through to the
last branch, so the "ClassifiedMap" was a single colour. The same run produced
the EVI panel spanning -5000..1000 because raw 16-bit digital numbers were fed
into the formula without the `+1` (L) term.

The current pipeline writes the same schema (`Class,PixelCount`) from a real
NDVI + red-edge classification: 5517 healthy / 3035 moderate / 664 stressed on
the default snapshot. See `matlab/index_panel.m` for the corrected MATLAB path.
