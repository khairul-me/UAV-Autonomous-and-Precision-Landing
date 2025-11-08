# System Verification Report

**Generated:** 2025-11-08T11:48:10.595740

---

## Summary

- **Total Tests:** 8
- **Passed:** 3 ✓
- **Failed:** 1 ✗
- **Skipped:** 4 ⊘
- **Partial:** 0 ◐
- **Success Rate:** 37.5%

---

## Detailed Results

### ❌ Airsim Connection

**Status:** FAILED

**Errors:**
- Retry connection over the limit

---

### ✅ Configuration

**Status:** PASSED

**Details:**
```json
{
  "settings_path": "C:\\Users\\Khairul\\Documents\\AirSim\\settings.json",
  "vehicles": [
    "Drone1"
  ],
  "num_cameras": 0,
  "num_sensors": 0
}
```

---

### ⊘ Multi Camera

**Status:** SKIPPED

**Errors:**
- AirSim connection unavailable

---

### ⊘ Feature Extraction

**Status:** SKIPPED

**Errors:**
- MultiCameraManager unavailable

---

### ⊘ Observation Builder

**Status:** SKIPPED

**Errors:**
- Prerequisite components unavailable

---

### ✅ Logging System

**Status:** PASSED

**Details:**
```json
{
  "episode_logger": "OK",
  "training_logger": "OK",
  "attack_logger": "OK"
}
```

---

### ⊘ Obstacle Generation

**Status:** SKIPPED

**Errors:**
- AirSim connection unavailable

---

### ✅ File Structure

**Status:** PASSED

**Details:**
```json
{
  "found": {
    "utils/multi_camera.py": {
      "size": 13617,
      "description": "Multi-camera manager"
    },
    "environments/observations.py": {
      "size": 14647,
      "description": "Observation builder"
    },
    "utils/episode_logger.py": {
      "size": 14941,
      "description": "Logging system"
    },
    "utils/training_monitor.py": {
      "size": 13120,
      "description": "Training monitor"
    },
    "environments/advanced_obstacles.py": {
      "size": 9962,
      "description": "Advanced obstacle generator"
    },
    "train_enhanced.py": {
      "size": 19246,
      "description": "Enhanced training script"
    }
  },
  "missing": [],
  "coverage": "6/6"
}
```

---

## Generated Artefacts

Artifacts saved in `verification_output/` include samples, obstacle layouts, and logs.
