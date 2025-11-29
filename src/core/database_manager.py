import sqlite3
import os

from datetime import datetime
from sqlite3 import Connection

class DatabaseManager:

    def __init__(self, path: str):
        self.path = path
        
        os.makedirs(os.path.dirname(path), exist_ok = True)
        self.create_table()

    def connect(self) -> Connection:
        connection = sqlite3.connect(self.path)
        return connection

    def create_table(self) -> None:
        connection = self.connect()

        connection.cursor().execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                run_number INTEGER NOT NULL,
                reprompt_info INTEGER NOT NULL,
                problem TEXT NOT NULL,
                diagram TEXT NOT NULL,
                level TEXT NOT NULL,
                test_type TEXT NOT NULL,
                input TEXT NOT NULL,
                expected TEXT NOT NULL,
                actual TEXT,
                passed BOOLEAN NOT NULL,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        connection.cursor().execute('''
            CREATE TABLE IF NOT EXISTS times (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                run_number INTEGER NOT NULL,
                time DOUBLE NOT NULL,
                gpu DOUBLE NOT NULL
            )
        ''')
        
        connection.commit()
        connection.close()
    
    def insert_results(self, results, model_name, run_number, reprompt_info, problem, diagram, level, test_type):
        conn = sqlite3.connect(self.path)
        cursor = conn.cursor()
        
        for idx, result in enumerate(results, start=1):
            cursor.execute('''
                INSERT INTO results 
                (model_name, run_number, reprompt_info, problem, diagram, level, test_type, 
                 input, expected, actual, passed, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_name,
                run_number,
                reprompt_info,
                problem,
                diagram,
                level,
                test_type,
                str(result['input']),
                str(result['expected']),
                str(result['actual']) if result['actual'] is not None else None,
                result['passed'],
                result.get('error'),
                datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()
    
    def insert_metrics(self, model_name, run_number, time, gpu):
        conn = sqlite3.connect(self.path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO times 
            (model_name, run_number, time, gpu)
            VALUES (?, ?, ?, ?)
        ''', (
            model_name,
            run_number,
            time,
            gpu
        ))
        
        conn.commit()
        conn.close()