"""
Pushover notification client for sending experimental results and alerts.

This module provides an optional interface to Pushover notifications.
If Pushover credentials are not configured or the library is not installed,
all operations will fail gracefully with warnings without terminating execution.
"""

import os
import warnings
from typing import Optional


class PushoverNotifier:
    """
    Object-oriented interface for sending notifications via Pushover.
    
    On instantiation, creates a client connection using app token and user key
    from environment variables. If credentials are missing or invalid, the
    client will operate in a no-op mode with warnings.
    
    Environment Variables Required:
        PUSHOVER_APP_TOKEN: Your Pushover application token
        PUSHOVER_USER_KEY: Your Pushover user key
    
    Example:
        >>> notifier = PushoverNotifier()
        >>> notifier.send_message(
        ...     "Experiment completed successfully!",
        ...     title="Training Complete",
        ...     priority=0
        ... )
    """
    
    def __init__(self):
        """
        Initialize the Pushover client.
        
        Reads credentials from environment variables. If credentials are missing
        or the pushover library is not available, the client will be disabled
        and operations will log warnings instead of failing.
        """
        self._pushover_client = None
        self._enabled = False
        
        # Check if pushover library is available
        try:
            from pushover import Pushover
            self._Pushover = Pushover
        except ImportError:
            warnings.warn(
                "python-pushover library not installed. Pushover notifications "
                "will be disabled. Install with: pip install python-pushover",
                UserWarning
            )
            return
        
        # Read credentials from environment variables
        app_token = os.getenv("PUSHOVER_APP_TOKEN")
        user_key = os.getenv("PUSHOVER_USER_KEY")
        
        if not app_token or not user_key:
            warnings.warn(
                "Pushover credentials not found in environment variables. "
                "Set PUSHOVER_APP_TOKEN and PUSHOVER_USER_KEY to enable "
                "notifications. Pushover notifications will be disabled.",
                UserWarning
            )
            return
        
        # Initialize the client
        try:
            self._pushover_client = self._Pushover(app_token)
            self._pushover_client.user(user_key)
            self._enabled = True
        except Exception as e:
            warnings.warn(
                f"Failed to initialize Pushover client: {e}. "
                "Pushover notifications will be disabled.",
                UserWarning
            )
    
    def send_message(
        self,
        message: str,
        title: Optional[str] = None,
        priority: int = 0,
        url: Optional[str] = None,
        url_title: Optional[str] = None,
    ) -> bool:
        """
        Send a notification message via Pushover.
        
        Args:
            message: The message body to send
            title: Optional title for the notification
            priority: Priority level (-2, -1, 0=normal, 1=high, 2=emergency)
            url: Optional URL to include in the notification
            url_title: Optional title for the URL link
        
        Returns:
            bool: True if message was sent successfully, False otherwise.
                  Always returns False if Pushover is not configured.
        
        Note:
            This method will never raise an exception. If Pushover is not
            configured or an error occurs, a warning will be logged and
            False will be returned.
        """
        if not self._enabled:
            warnings.warn(
                "Pushover notifications are disabled. Message not sent.",
                UserWarning
            )
            return False
        
        if self._pushover_client is None:
            warnings.warn(
                "Pushover client not initialized. Message not sent.",
                UserWarning
            )
            return False

        if title is None:
            machine_name = os.getenv("MACHINE_NAME", "Unknown Machine") 
            title = f"Training Report on {machine_name}"
        
        try:
            # Validate priority range
            if priority not in [-2, -1, 0, 1, 2]:
                warnings.warn(
                    f"Invalid priority level {priority}. Must be one of "
                    "[-2, -1, 0, 1, 2]. Using default priority 0.",
                    UserWarning
                )
                priority = 0
            
            # Create message object
            msg = self._pushover_client.msg(message)
            
            # Set message properties using .set() method
            if title is not None:
                msg.set("title", title)
            if priority != 0:
                msg.set("priority", priority)
            if url is not None:
                msg.set("url", url)
            if url_title is not None:
                msg.set("url_title", url_title)
            
            # Send the message
            self._pushover_client.send(msg)
            return True
            
        except Exception as e:
            warnings.warn(
                f"Failed to send Pushover notification: {e}. "
                "Message not sent.",
                UserWarning
            )
            return False
    
    def is_enabled(self) -> bool:
        """
        Check if Pushover notifications are enabled and configured.
        
        Returns:
            bool: True if Pushover is properly configured and ready to send
                  messages, False otherwise.
        """
        return self._enabled and self._pushover_client is not None
