# Copyright Reexpress AI, Inc. All rights reserved.

import sqlite3
from typing import Optional, Dict, Any
from contextlib import contextmanager
import html


class DocumentDatabase:
    """A class to manage documents in an SQLite database."""

    def __init__(self, db_path: str = "documents.db"):
        """
        Initialize the DocumentDatabase.

        Args:
            db_path: Path to the SQLite database file (default: "documents.db")
        """
        self.db_path = db_path
        self.create_table()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()

    def create_table(self):
        """
        Create the documents table if it doesn't exist.

        -- Model field mapping (v1.2.0+):
        -- model1_* fields: GPT-5 (originally model4)
        -- model2_* fields: Gemini 2.5 Pro (originally model3)
        -- model3_* fields: Pre-training model (originally model1)
        -- model4_* fields: Pre-training model (originally model2)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    model1_summary TEXT,
                    model1_explanation TEXT,
                    model2_explanation TEXT,
                    model3_explanation TEXT,
                    model4_explanation TEXT,
                    model1_classification_int INTEGER,
                    model2_classification_int INTEGER,
                    model3_classification_int INTEGER,
                    model4_classification_int INTEGER,
                    agreement_model_classification_int INTEGER,
                    label_int INTEGER,
                    label_was_updated_int INTEGER,
                    document_source TEXT,
                    info TEXT,
                    user_question TEXT,
                    ai_response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create trigger to automatically update updated_at on row update
            cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS update_documents_timestamp 
                AFTER UPDATE ON documents
                FOR EACH ROW
                BEGIN
                    UPDATE documents SET updated_at = CURRENT_TIMESTAMP 
                    WHERE document_id = NEW.document_id;
                END
            ''')

            # Create index on timestamps for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON documents(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_updated_at ON documents(updated_at)')

            conn.commit()

    def add_document(self, document_id: str, model1_summary: str, model1_explanation: str,
                     model2_explanation: str, model3_explanation: str,
                     model4_explanation: str,
                     model1_classification_int: int, model2_classification_int: int,
                     model3_classification_int: int, model4_classification_int: int,
                     agreement_model_classification_int: int,
                     label_int: int, label_was_updated_int: int, document_source: str, info: str,
                     user_question: str, ai_response: str) -> bool:
        """
        Add a new document to the database. See create_table() for the model name semantics.

        Args:
            document_id: Unique identifier for the document
            model1_summary: Short summary from model 1
            model1_explanation: Explanation from model 1
            model2_explanation: Explanation from model 2
            model3_explanation: Explanation from model 3
            model4_explanation: Explanation from model 4
            model1_classification_int: Classification result from model 1
            model2_classification_int: Classification result from model 2
            model3_classification_int: Classification result from model 3
            model4_classification_int: Classification result from model 4
            agreement_model_classification_int: Agreement classification from the agreement model
            label_int: Ground-truth label
            label_was_updated_int: Indicator if label was updated
            document_source: Data source
            info: Additional metadata
            user_question: The user's question
            ai_response: The AI's response

        Returns:
            True if successful, False if document_id already exists
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO documents (
                        document_id, model1_summary, model1_explanation, model2_explanation,
                        model3_explanation, model4_explanation, model1_classification_int, model2_classification_int,
                        model3_classification_int, model4_classification_int, agreement_model_classification_int,
                        label_int, label_was_updated_int, document_source, info,
                        user_question, ai_response
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (document_id, model1_summary, model1_explanation, model2_explanation, model3_explanation,
                      model4_explanation,
                      model1_classification_int, model2_classification_int,
                      model3_classification_int, model4_classification_int, agreement_model_classification_int, label_int,
                      label_was_updated_int, document_source, info,
                      user_question, ai_response))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            # Document ID already exists
            return False

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID.

        Args:
            document_id: The unique identifier of the document

        Returns:
            Dictionary containing document data with html escaped, or None if not found
            (Note that we also currently escape the document_id)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM documents WHERE document_id = ?
            ''', (document_id,))

            row = cursor.fetchone()
            if row:
                row_dict = dict(row)
                row_dict_for_display = {}
                # Escape HTML content for safe display
                for key in row_dict:
                    # Don't escape integer fields
                    if key in ['model1_classification_int', 'model2_classification_int',
                               'model3_classification_int', 'model4_classification_int',
                               'agreement_model_classification_int',
                               'label_int', 'label_was_updated_int']:
                        row_dict_for_display[key] = row_dict[key]
                    else:
                        row_dict_for_display[key] = html.escape(str(row_dict[key]))
                return row_dict_for_display
            return None

    def update_document(self, document_id: str, **kwargs) -> bool:
        """
        Update specific fields of a document.
        Note: updated_at will be automatically updated by the trigger.

        Args:
            document_id: The unique identifier of the document
            **kwargs: Field names and their new values

        Returns:
            True if update was successful, False if document not found
        """
        # Filter out invalid field names
        valid_fields = {'model1_summary', 'model1_explanation', 'model2_explanation',
                        'model3_explanation', 'model4_explanation', 'model1_classification_int',
                        'model2_classification_int', 'model3_classification_int',
                        'model4_classification_int',
                        'agreement_model_classification_int', 'label_int', 'label_was_updated_int',
                        'document_source', 'info', 'user_question', 'ai_response'}
        fields_to_update = {k: v for k, v in kwargs.items() if k in valid_fields}

        if not fields_to_update:
            return False

        # Build the UPDATE query
        set_clause = ', '.join([f"{field} = ?" for field in fields_to_update])
        values = list(fields_to_update.values()) + [document_id]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                UPDATE documents 
                SET {set_clause}
                WHERE document_id = ?
            ''', values)
            conn.commit()

            return cursor.rowcount > 0

    def document_exists(self, document_id: str) -> bool:
        """
        Check if a document exists in the database.

        Args:
            document_id: The unique identifier to check

        Returns:
            True if document exists, False otherwise
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM documents WHERE document_id = ?
            ''', (document_id,))
            return cursor.fetchone()[0] > 0

    def get_document_count(self) -> int:
        """
        Get the total number of documents in the database.

        Returns:
            The count of documents
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM documents')
            return cursor.fetchone()[0]

    def get_recent_documents(self, limit: int = 10) -> list[Dict[str, Any]]:
        """
        Get the most recently created documents.

        Args:
            limit: Maximum number of documents to return (default: 10)

        Returns:
            List of dictionaries containing recent documents with HTML escaped
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM documents 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))

            documents = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                row_dict_for_display = {}
                # Escape HTML content for safe display
                for key in row_dict:
                    # Don't escape integer fields
                    if key in ['model1_classification_int', 'model2_classification_int',
                               'model3_classification_int', 'model4_classification_int',
                               'agreement_model_classification_int',
                               'label_int', 'label_was_updated_int']:
                        row_dict_for_display[key] = row_dict[key]
                    else:
                        row_dict_for_display[key] = html.escape(str(row_dict[key]))
                documents.append(row_dict_for_display)

            return documents

    def get_recently_updated_documents(self, limit: int = 10) -> list[Dict[str, Any]]:
        """
        Get the most recently updated documents.

        Args:
            limit: Maximum number of documents to return (default: 10)

        Returns:
            List of dictionaries containing recently updated documents with HTML escaped
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM documents 
                ORDER BY updated_at DESC 
                LIMIT ?
            ''', (limit,))

            documents = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                row_dict_for_display = {}
                # Escape HTML content for safe display
                for key in row_dict:
                    # Don't escape integer fields
                    if key in ['model1_classification_int', 'model2_classification_int',
                               'model3_classification_int', 'model4_classification_int',
                               'agreement_model_classification_int',
                               'label_int', 'label_was_updated_int']:
                        row_dict_for_display[key] = row_dict[key]
                    else:
                        row_dict_for_display[key] = html.escape(str(row_dict[key]))
                documents.append(row_dict_for_display)

            return documents


# Example usage
if __name__ == "__main__":
    # Create database instance
    db = DocumentDatabase("/Users/a/Documents/temp_delete_any_time/debug/example_docs.db")

    # Add a document
    success = db.add_document(
        document_id="DOC001",
        model1_summary="A brief summary from Model 1",
        model1_explanation="Model 1 thinks this is correct",
        model2_explanation="Model 2 agrees with high confidence",
        model3_explanation="Model 3 has some reservations",
        model4_explanation="Model 4 is unsure",
        model1_classification_int=1,
        model2_classification_int=1,
        model3_classification_int=0,
        model4_classification_int=0,
        agreement_model_classification_int=1,
        label_int=1,
        label_was_updated_int=0,
        document_source="open source",
        info="hf",
        user_question="What is the capital of France?",
        ai_response="The capital of France is Paris."
    )
    print(f"Document added: {success}")

    success = db.add_document(
        document_id="DOC002",
        model1_summary="aA brief summary from Model 1",
        model1_explanation="aModel 1 thinks this is correct",
        model2_explanation="aModel 2 agrees with high confidence",
        model3_explanation="aModel 3 has some reservations",
        model4_explanation="aModel 4 is unsure",
        model1_classification_int=1,
        model2_classification_int=0,
        model3_classification_int=1,
        model4_classification_int=0,
        agreement_model_classification_int=0,
        label_int=1,
        label_was_updated_int=1,
        document_source="open source",
        info="hf",
        user_question="aWhat is the capital of France?",
        ai_response="aThe capital of France is Paris."
    )
    print(f"Document added: {success}")

    success = db.add_document(
        document_id="DOC003",
        model1_summary="aA brief summary from Model 1",
        model1_explanation="aModel 1 thinks this is correct",
        model2_explanation="aModel 2 agrees with high confidence",
        model3_explanation="aModel 3 has some reservations",
        model4_explanation="aModel 4 is unsure",
        model1_classification_int=1,
        model2_classification_int=1,
        model3_classification_int=1,
        model4_classification_int=1,
        agreement_model_classification_int=1,
        label_int=1,
        label_was_updated_int=0,
        document_source="open source",
        info="hf",
        user_question="<a href='https://re.express/'></a>",
        ai_response="aThe capital of France is Paris."
    )
    print(f"Document added: {success}")

    # Retrieve a document
    doc = db.get_document("DOC001")
    if doc:
        print(f"Retrieved document: {doc}")

    doc = db.get_document("DOC003")
    if doc:
        print(f"Retrieved document: {doc}")

    # Update a document
    updated = db.update_document(
        "DOC001",
        model1_explanation="Model 1 is very confident",
        model1_classification_int=0
    )
    print(f"Document updated: {updated}")

    updated = db.update_document(
        "DOC001",
        model1_explanation="Model 1 is very confident",
        ai_response="The capital of France is Paris, a country in Europe.",
        model2_classification_int=0,
        label_was_updated_int=1
    )
    print(f"Document updated: {updated}")

    # Check if document exists
    exists = db.document_exists("DOC001")
    print(f"Document exists: {exists}")

    # Get document count
    count = db.get_document_count()
    print(f"Total documents: {count}")

    # Get recent documents
    recent_docs = db.get_recent_documents(5)
    print(f"Recent documents: {len(recent_docs)}")

    # Get recently updated documents
    updated_docs = db.get_recently_updated_documents(5)
    print(f"Recently updated documents: {len(updated_docs)}")
