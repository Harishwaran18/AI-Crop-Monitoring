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

Changes made while archiving:

* fixed the `SyntaxError` in `sms_streamlit.py` (the recipient is interpolated now)
* removed committed phone numbers and Twilio placeholders; everything reads from
  environment variables (`ALERT_RECIPIENT`, `TWILIO_*`, `FAST2SMS_API_KEY`)
* replaced the hardcoded `C:\Users\...` path in `plant_demo.py`

Known limitations that motivated the rewrite: untrained models presented as
predictions, random "multispectral" channels that were discarded before use, no
caching, and duplicated label sets across apps.
