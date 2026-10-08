-- ==============================================================================
-- Real-Time Object Detection Platform - Database Schema
-- Database: vision_platform
-- ==============================================================================

CREATE DATABASE IF NOT EXISTS vision_platform;
USE vision_platform;

CREATE TABLE IF NOT EXISTS detection_logs (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    object_class VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    bbox_x INT NOT NULL,
    bbox_y INT NOT NULL,
    bbox_w INT NOT NULL,
    bbox_h INT NOT NULL,
    INDEX idx_timestamp (timestamp),
    INDEX idx_object_class (object_class)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
