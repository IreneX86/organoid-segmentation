"""Use exactly the previous train-only BN procedure for the interline ablation."""
from pathlib import Path
import er_roi_bn_recalibrate as calibration
R=Path(__file__).resolve().parents[1]
calibration.DATA=R/'outputs/er_interline_student_ready_20261029'
calibration.SRC=R/'outputs/er_interline_student_run_20261029/extracted/results/student'
calibration.OUT=R/'outputs/er_interline_bn_calibrated_20261029'
if __name__=='__main__':calibration.main()
