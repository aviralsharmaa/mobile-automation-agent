"""
Ngrok Tunnel - ADB port forwarding for remote access
"""
import subprocess
import time
import requests
from typing import Optional, Dict
from src.utils.logging import logger


class NgrokTunnel:
    """Manage ngrok tunnel for ADB"""
    
    def __init__(self, auth_token: Optional[str] = None, port: int = 5555):
        """
        Initialize ngrok tunnel
        
        Args:
            auth_token: Ngrok auth token (optional but recommended)
            port: ADB TCP port to tunnel (default: 5555)
        """
        self.auth_token = auth_token
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.tunnel_url: Optional[str] = None
    
    def start(self) -> bool:
        """
        Start ngrok tunnel
        
        Returns:
            True if successful
        """
        try:
            # First, set up ADB to listen on TCP port
            subprocess.run(["adb", "tcpip", str(self.port)], check=True, capture_output=True)
            logger.info(f"ADB listening on TCP port {self.port}")
            
            # Start ngrok
            cmd = ["ngrok", "tcp", str(self.port)]
            if self.auth_token:
                cmd.extend(["--authtoken", self.auth_token])
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for ngrok to start
            time.sleep(2)
            
            # Get tunnel URL
            self.tunnel_url = self._get_tunnel_url()
            
            if self.tunnel_url:
                logger.info(f"Ngrok tunnel started: {self.tunnel_url}")
                return True
            else:
                logger.error("Failed to get ngrok tunnel URL")
                return False
                
        except FileNotFoundError:
            logger.error("ngrok not found. Please install ngrok: https://ngrok.com/download")
            return False
        except Exception as e:
            logger.error(f"Error starting ngrok tunnel: {e}")
            return False
    
    def _get_tunnel_url(self) -> Optional[str]:
        """Get ngrok tunnel URL from API"""
        try:
            response = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=5)
            if response.status_code == 200:
                data = response.json()
                tunnels = data.get("tunnels", [])
                if tunnels:
                    public_url = tunnels[0].get("public_url", "")
                    # Extract host and port
                    # Format: tcp://0.tcp.ngrok.io:12345
                    return public_url.replace("tcp://", "")
            return None
        except Exception as e:
            logger.error(f"Error getting tunnel URL: {e}")
            return None
    
    def stop(self):
        """Stop ngrok tunnel"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
            logger.info("Ngrok tunnel stopped")
    
    def get_connection_string(self) -> Optional[str]:
        """
        Get connection string for remote ADB connection
        
        Returns:
            Connection string (host:port) or None
        """
        return self.tunnel_url
    
    def connect_remote(self) -> bool:
        """
        Connect to remote device via ngrok
        
        Returns:
            True if connection successful
        """
        if not self.tunnel_url:
            logger.error("Tunnel not started")
            return False
        
        try:
            result = subprocess.run(
                ["adb", "connect", self.tunnel_url],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Connected to remote device: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to connect: {e.stderr}")
            return False


def setup_ngrok_tunnel(auth_token: Optional[str] = None, port: int = 5555) -> Optional[NgrokTunnel]:
    """
    Setup ngrok tunnel for ADB
    
    Args:
        auth_token: Ngrok auth token
        port: ADB TCP port
        
    Returns:
        NgrokTunnel instance or None if failed
    """
    tunnel = NgrokTunnel(auth_token=auth_token, port=port)
    if tunnel.start():
        return tunnel
    return None
