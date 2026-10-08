"""
Database management module for the Real-Time Object Detection Platform.
Handles MySQL connections, schema setup, logging detection events,
and querying analytics with resilient local fallback.
"""

import datetime
import logging
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
    MYSQL_AVAILABLE = True
except ImportError:
    mysql = None
    MySQLError = Exception
    MYSQL_AVAILABLE = False

from config import DatabaseConfig, db_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connectivity and logging for object detection events."""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or db_config
        self.backend: str = "disconnected"
        self.last_error: Optional[str] = None
        self._init_connection()

    def _init_connection(self) -> None:
        """Attempt connecting to MySQL; fall back to SQLite if configured."""
        self.last_error = None

        if MYSQL_AVAILABLE:
            try:
                # Step 1: Check server connection & ensure database exists
                server_conn = mysql.connector.connect(
                    host=self.config.host,
                    port=self.config.port,
                    user=self.config.user,
                    password=self.config.password,
                    connect_timeout=3
                )
                if server_conn.is_connected():
                    cursor = server_conn.cursor()
                    cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{self.config.database}`;")
                    cursor.close()
                    server_conn.close()

                # Step 2: Connect to specific database
                conn = self._get_mysql_connection()
                if conn and conn.is_connected():
                    self.backend = "mysql"
                    self._create_mysql_tables(conn)
                    conn.close()
                    logger.info("Successfully connected to MySQL database: %s", self.config.database)
                    return
            except Exception as exc:
                self.last_error = f"MySQL connection failed: {exc}"
                logger.warning("Could not connect to MySQL: %s", exc)

        # Step 3: Handle fallback if MySQL connection failed or not available
        if self.config.fallback_sqlite:
            try:
                self._create_sqlite_tables()
                self.backend = "sqlite"
                logger.info("Activated SQLite fallback database: %s", self.config.sqlite_file)
            except Exception as sqlite_exc:
                self.backend = "disconnected"
                self.last_error = f"SQLite fallback failed: {sqlite_exc}"
                logger.error("Failed to initialize SQLite fallback: %s", sqlite_exc)
        else:
            self.backend = "disconnected"

    def _get_mysql_connection(self):
        """Create a fresh MySQL connection."""
        if not MYSQL_AVAILABLE:
            return None
        return mysql.connector.connect(
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
            connect_timeout=3
        )

    def _get_sqlite_connection(self):
        """Create a fresh SQLite connection."""
        return sqlite3.connect(str(self.config.sqlite_file), timeout=5)

    def _create_mysql_tables(self, conn) -> None:
        """Create MySQL table if it does not already exist."""
        cursor = conn.cursor()
        create_query = """
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
        """
        cursor.execute(create_query)
        conn.commit()
        cursor.close()

    def _create_sqlite_tables(self) -> None:
        """Create SQLite table if it does not already exist."""
        with self._get_sqlite_connection() as conn:
            cursor = conn.cursor()
            create_query = """
            CREATE TABLE IF NOT EXISTS detection_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                object_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                bbox_x INTEGER NOT NULL,
                bbox_y INTEGER NOT NULL,
                bbox_w INTEGER NOT NULL,
                bbox_h INTEGER NOT NULL
            );
            """
            cursor.execute(create_query)
            conn.commit()

    def is_connected(self) -> bool:
        """Return True if a database backend is active."""
        return self.backend in ("mysql", "sqlite")

    def get_status(self) -> Dict[str, Any]:
        """Return connectivity metadata for UI display."""
        return {
            "connected": self.is_connected(),
            "backend": self.backend,
            "host": self.config.host if self.backend == "mysql" else "Local SQLite File",
            "database": self.config.database if self.backend == "mysql" else self.config.sqlite_file.name,
            "last_error": self.last_error
        }

    def reconnect(self) -> Dict[str, Any]:
        """Force reconnect attempt."""
        self._init_connection()
        return self.get_status()

    def log_detection(
        self,
        object_class: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        timestamp: Optional[datetime.datetime] = None
    ) -> bool:
        """
        Log a single detection event to the active database.
        bbox format: (x, y, w, h)
        """
        return self.log_detections_batch([
            {
                "object_class": object_class,
                "confidence": confidence,
                "bbox": bbox,
                "timestamp": timestamp or datetime.datetime.now()
            }
        ]) > 0

    def log_detections_batch(self, detections: List[Dict[str, Any]]) -> int:
        """
        Log a list of detections in batch.
        Each item must contain: object_class, confidence, bbox (x, y, w, h), and optional timestamp.
        Returns number of logged records.
        """
        if not detections or not self.is_connected():
            return 0

        rows = []
        for d in detections:
            bbox = d.get("bbox", (0, 0, 0, 0))
            ts = d.get("timestamp") or datetime.datetime.now()
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            rows.append((
                ts_str,
                str(d["object_class"]),
                float(d["confidence"]),
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3])
            ))

        try:
            if self.backend == "mysql":
                conn = self._get_mysql_connection()
                cursor = conn.cursor()
                query = """
                INSERT INTO detection_logs
                (timestamp, object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                cursor.executemany(query, rows)
                conn.commit()
                cursor.close()
                conn.close()
                return len(rows)

            elif self.backend == "sqlite":
                with self._get_sqlite_connection() as conn:
                    cursor = conn.cursor()
                    query = """
                    INSERT INTO detection_logs
                    (timestamp, object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """
                    cursor.executemany(query, rows)
                    conn.commit()
                    return len(rows)

        except Exception as exc:
            logger.error("Failed to insert detection logs: %s", exc)
            self.last_error = str(exc)
            return 0

        return 0

    def fetch_recent_logs(self, limit: int = 50, filter_class: Optional[str] = None) -> pd.DataFrame:
        """Fetch the most recent detection events as a Pandas DataFrame."""
        columns = ["log_id", "timestamp", "object_class", "confidence", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
        if not self.is_connected():
            return pd.DataFrame(columns=columns)

        try:
            query = "SELECT log_id, timestamp, object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h FROM detection_logs"
            params = []

            if filter_class and filter_class.lower() != "all":
                query += " WHERE object_class = ?" if self.backend == "sqlite" else " WHERE object_class = %s"
                params.append(filter_class)

            query += " ORDER BY log_id DESC LIMIT ?" if self.backend == "sqlite" else " ORDER BY log_id DESC LIMIT %s"
            params.append(limit)

            if self.backend == "mysql":
                conn = self._get_mysql_connection()
                df = pd.read_sql_query(query, conn, params=params)
                conn.close()
                return df
            elif self.backend == "sqlite":
                with self._get_sqlite_connection() as conn:
                    df = pd.read_sql_query(query, conn, params=params)
                    return df

        except Exception as exc:
            logger.error("Failed to fetch logs: %s", exc)
            self.last_error = str(exc)
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(columns=columns)

    def fetch_metrics(self) -> Dict[str, Any]:
        """Calculate live metrics for the UI dashboard."""
        metrics = {
            "total_detections": 0,
            "unique_classes": 0,
            "top_object": "None",
            "class_distribution": pd.DataFrame(columns=["object_class", "count"])
        }

        if not self.is_connected():
            return metrics

        try:
            if self.backend == "mysql":
                conn = self._get_mysql_connection()
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*), COUNT(DISTINCT object_class) FROM detection_logs")
                total, unique = cur.fetchone() or (0, 0)
                metrics["total_detections"] = total
                metrics["unique_classes"] = unique

                dist_query = """
                SELECT object_class, COUNT(*) as count 
                FROM detection_logs 
                GROUP BY object_class 
                ORDER BY count DESC 
                LIMIT 10
                """
                dist_df = pd.read_sql_query(dist_query, conn)
                metrics["class_distribution"] = dist_df
                if not dist_df.empty:
                    metrics["top_object"] = f"{dist_df.iloc[0]['object_class']} ({dist_df.iloc[0]['count']})"
                cur.close()
                conn.close()

            elif self.backend == "sqlite":
                with self._get_sqlite_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT COUNT(*), COUNT(DISTINCT object_class) FROM detection_logs")
                    total, unique = cur.fetchone() or (0, 0)
                    metrics["total_detections"] = total
                    metrics["unique_classes"] = unique

                    dist_query = """
                    SELECT object_class, COUNT(*) as count 
                    FROM detection_logs 
                    GROUP BY object_class 
                    ORDER BY count DESC 
                    LIMIT 10
                    """
                    dist_df = pd.read_sql_query(dist_query, conn)
                    metrics["class_distribution"] = dist_df
                    if not dist_df.empty:
                        metrics["top_object"] = f"{dist_df.iloc[0]['object_class']} ({dist_df.iloc[0]['count']})"

        except Exception as exc:
            logger.error("Failed to fetch metrics: %s", exc)

        return metrics

    def clear_logs(self) -> bool:
        """Clear all records from detection_logs."""
        if not self.is_connected():
            return False

        try:
            if self.backend == "mysql":
                conn = self._get_mysql_connection()
                cur = conn.cursor()
                cur.execute("TRUNCATE TABLE detection_logs")
                conn.commit()
                cur.close()
                conn.close()
                return True
            elif self.backend == "sqlite":
                with self._get_sqlite_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("DELETE FROM detection_logs")
                    cur.execute("DELETE FROM sqlite_sequence WHERE name='detection_logs'")
                    conn.commit()
                    return True
        except Exception as exc:
            logger.error("Failed to clear logs: %s", exc)
            return False

        return False
