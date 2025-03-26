#!/usr/bin/env python3
"""
MQTT communication handler for SpectrumAlert.
"""

import json
import os
import time
from typing import Dict, Tuple, Optional, Any

import paho.mqtt.client as mqtt


class MQTTHandler:
    """Handles MQTT communication."""
    
    def __init__(self):
        """Initialize MQTT Handler."""
        self.client = None
        self.topic = None
        
    def setup_client(self) -> Tuple[Optional[mqtt.Client], Optional[str]]:
        """
        Initializes and configures the MQTT client using environment variables.
        
        Returns:
            Tuple of (mqtt_client, mqtt_topic) or (None, None) if setup fails
        """
        try:
            # Load environment variables
            broker = os.getenv("MQTT_BROKER", "localhost")
            port = int(os.getenv("MQTT_PORT", 1883))
            user = os.getenv("MQTT_USER", None)
            password = os.getenv("MQTT_PASSWORD", None)
            topic = os.getenv("MQTT_TOPIC", "spectrum/anomaly")
            
            # TLS & CA Certificate Options
            use_tls = int(os.getenv("MQTT_TLS", 0))
            use_ca_cert = int(os.getenv("MQTT_USE_CA_CERT", 0))
            ca_cert_path = os.getenv("MQTT_CA_CERT", "/path/to/ca.crt")
            
            print(f"📡 Configuring MQTT: {broker}:{port} (TLS: {use_tls}, CA Cert: {use_ca_cert})")
            
            # Create MQTT client
            mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            
            # Enable automatic reconnect
            mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)
            
            # Configure TLS if enabled
            if use_tls:
                print("🔐 Enabling TLS for MQTT...")
                mqtt_client.tls_set(ca_certs=ca_cert_path if use_ca_cert else None)
            
            # Set callbacks
            mqtt_client.on_connect = self._on_connect
            mqtt_client.on_disconnect = self._on_disconnect
            
            # Set authentication if provided
            if user or password:
                mqtt_client.username_pw_set(user, password)
            
            # Connect to broker
            mqtt_client.connect(broker, port, 60)
            print("✅ Connected to MQTT broker successfully!")
            
            self.client = mqtt_client
            self.topic = topic
            return mqtt_client, topic
            
        except Exception as e:
            print(f"❌ MQTT Setup Error: {e}")
            return None, None
    
    def publish_message(self, topic: str, payload: Dict[str, Any]) -> bool:
        """
        Publishes a message to the MQTT broker.
        
        Args:
            topic: MQTT topic to publish to
            payload: Dictionary with message data
            
        Returns:
            Boolean indicating success
        """
        if not self.client:
            print("❌ MQTT client not initialized")
            return False
            
        try:
            payload_str = json.dumps(payload)
            publish_info = self.client.publish(topic, payload_str)
            
            if publish_info.rc is not None:
                publish_info.wait_for_publish(timeout=10)
                print(f"📤 Published to MQTT topic '{topic}'")
                return True
            else:
                print(f"⚠️ MQTT Publish failed: No response received.")
                return False
                
        except Exception as e:
            print(f"❌ MQTT Publishing Error: {e}")
            print(f"🔄 Trying to re-establish MQTT connection...")
            try:
                self.client.reconnect()
                time.sleep(2)  # Allow time for reconnection
                return False
            except Exception as recon_error:
                print(f"⚠️ MQTT Reconnect Failed: {recon_error}")
                return False
    
    def disconnect(self):
        """Disconnect the MQTT client if connected."""
        if self.client:
            try:
                self.client.disconnect()
                print("MQTT client disconnected")
            except Exception as e:
                print(f"❌ Error disconnecting MQTT: {e}")
    
    @staticmethod
    def _on_connect(client, userdata, flags, rc, properties):
        """Callback for MQTT connection events."""
        if rc == 0:
            print("✅ MQTT Connected Successfully!")
        else:
            print(f"⚠️ MQTT Connection Failed with Code {rc}")
    
    @staticmethod
    def _on_disconnect(client, userdata, rc, *args):
        """Callback for MQTT disconnection events."""
        print("❌ MQTT on_disconnect! Trying to reconnect...")
        try:
            client.reconnect()
        except Exception as e:
            print(f"⚠️ MQTT Reconnect Failed: {e}")
