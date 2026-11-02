"""
Google Drive Cloud Integration Module.
Provides secure, non-destructive synchronization of approved decoration photos to Google Drive.
Credentials are loaded strictly via environment variables or specified configuration files.
Never deletes original or remote files.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class GoogleDriveIntegration:
    def __init__(self, credentials_path: Optional[str] = None, destination_folder_id: Optional[str] = None):
        self.credentials_path = credentials_path or os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON")
        self.destination_folder_id = destination_folder_id or os.getenv("GDRIVE_FOLDER_ID")
        self.service = None
        self.is_connected = False
        self._authenticate()

    def _authenticate(self):
        """Attempts authentication using Google API client libraries if available."""
        if not self.credentials_path or not os.path.exists(self.credentials_path):
            logger.info("Google Drive: No credentials file provided. Operating in unconfigured state.")
            self.is_connected = False
            return

        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            creds = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=["https://www.googleapis.com/auth/drive.file"]
            )
            self.service = build("drive", "v3", credentials=creds)
            self.is_connected = True
            logger.info("Google Drive: Successfully authenticated via service account.")
        except Exception as e:
            logger.warning(f"Google Drive: Authentication setup failed ({e}).")
            self.service = None
            self.is_connected = False

    def test_connection(self) -> Tuple[bool, str]:
        """Validates connection and verifies permissions to the destination folder."""
        if not self.is_connected or self.service is None:
            return False, "Not authenticated. Please provide valid Service Account JSON credentials."

        if not self.destination_folder_id:
            return False, "Destination Folder ID is not configured."

        try:
            folder = self.service.files().get(fileId=self.destination_folder_id, fields="id, name").execute()
            return True, f"Connected to Google Drive folder: '{folder.get('name')}'"
        except Exception as e:
            return False, f"Failed accessing folder '{self.destination_folder_id}': {str(e)}"

    def upload_approved_image(self, local_filepath: str, subfolder_name: Optional[str] = None) -> Tuple[bool, str]:
        """
        Uploads an approved image to the configured Google Drive destination folder.
        Read-only on local; never deletes remote.
        """
        if not self.is_connected or self.service is None:
            return False, "Google Drive service is not connected."

        if not os.path.exists(local_filepath):
            return False, f"Local file not found: {local_filepath}"

        try:
            from googleapiclient.http import MediaFileUpload

            filename = os.path.basename(local_filepath)
            file_metadata = {
                "name": filename,
                "parents": [self.destination_folder_id]
            }
            media = MediaFileUpload(local_filepath, resumable=True)
            uploaded_file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id, name, webViewLink"
            ).execute()

            link = uploaded_file.get("webViewLink", "")
            return True, f"Successfully uploaded '{filename}' to Google Drive. Link: {link}"
        except Exception as e:
            logger.error(f"Google Drive upload error for {local_filepath}: {e}")
            return False, str(e)
