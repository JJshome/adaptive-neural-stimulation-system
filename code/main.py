# adaptive-neural-stimulation-system/code/main.py
import time
import os
from stimulation_module import StimulationModule, NeuralRegenerationPattern, StimulationMode
from logger_config import setup_logging

# Setup logging for the main script
logger = setup_logging()

def main():
    logger.info("Starting Adaptive Neural Stimulation System Demonstration.")

    # Create a directory for protocols if it doesn't exist
    os.makedirs("stimulation_protocols", exist_ok=True)

    # Initialize StimulationModule with 2 channels
    system = StimulationModule(n_channels=2, hardware_connected=False, visualization_enabled=True)

    # --- Demo 1: Apply a neural regeneration pattern and start stimulation ---
    logger.info("\n--- Demo 1: Apply Regeneration Pattern (Channel 0) & Start ---")
    if system.apply_regeneration_pattern(0, NeuralRegenerationPattern.EARLY_STAGE):
        logger.info("Early Stage pattern applied to Channel 0.")
    else:
        logger.error("Failed to apply pattern to Channel 0.")
        
    logger.info("\nStarting stimulation on Channel 0 for 10 seconds...")
    system.start_stimulation(channel_id=0, duration=10)
    time.sleep(3) # Let it run for a bit

    # --- Demo 2: Apply a different pattern to another channel and start ---
    logger.info("\n--- Demo 2: Apply Different Pattern (Channel 1) & Start ---")
    if system.apply_regeneration_pattern(1, NeuralRegenerationPattern.MOTOR_NERVE):
        logger.info("Motor Nerve pattern applied to Channel 1.")
    else:
        logger.error("Failed to apply pattern to Channel 1.")

    logger.info("\nStarting stimulation on Channel 1 indefinitely...")
    system.start_stimulation(channel_id=1)
    time.sleep(3) # Let it run for a bit

    # --- Demo 3: Change parameters mid-stimulation (Channel 1) ---
    logger.info("\n--- Demo 3: Change Parameters Mid-Stimulation (Channel 1) ---")
    logger.info("\nChanging Channel 1 frequency to 150Hz and amplitude to 3.0mA...")
    if system.set_parameters(1, {"frequency_hz": 150, "amplitude_mA": 3.0}):
        logger.info("Channel 1 parameters updated successfully.")
    else:
        logger.error("Failed to update Channel 1 parameters.")
    time.sleep(3)

    # --- Demo 4: Safety Limit Violation Test ---
    logger.info("\n--- Demo 4: Safety Limit Violation Test (Channel 0) ---")
    logger.info("\nAttempting to set unsafe amplitude for Channel 0 (should fail)...")
    if not system.set_parameters(0, {"amplitude_mA": 15.0}): # Max is 10.0 mA
        logger.info("Successfully blocked unsafe amplitude setting for Channel 0.")
    else:
        logger.error("UNSAFE: Unsafe amplitude was set for Channel 0!")
    time.sleep(1)

    logger.info("\nAttempting to set unsafe charge density for Channel 0 (should fail)...")
    # This combination (10mA * 6000us) results in 0.06 mC, which is within the default 0.5mC limit.
    # To make it unsafe, let's try a value that exceeds 0.5mC, e.g., 100mA * 6000us = 0.6 mC
    if not system.set_parameters(0, {"amplitude_mA": 100.0, "pulse_width_us": 6000.0}):
        logger.info("Successfully blocked unsafe charge density for Channel 0.")
    else:
        logger.error("UNSAFE: Unsafe charge density was set for Channel 0!")
    time.sleep(1)

    # --- Demo 5: Stop specific channel ---
    logger.info("\n--- Demo 5: Stop Specific Channel (Channel 1) ---")
    logger.info("\nStopping stimulation on Channel 1...")
    system.stop_stimulation(channel_id=1)
    time.sleep(2)

    # --- Demo 6: Save and Load Protocol ---
    logger.info("\n--- Demo 6: Save and Load Protocol ---")
    protocol_filepath = "stimulation_protocols/my_test_protocol.json"
    logger.info(f"\nSaving current protocol to {protocol_filepath}...")
    if system.save_protocol(protocol_filepath):
        logger.info("Protocol saved.")
    else:
        logger.error("Failed to save protocol.")
    
    # Change Channel 0 parameters to something else for testing load
    system.set_parameters(0, {"amplitude_mA": 0.1, "frequency_hz": 1})
    logger.info("\nChannel 0 parameters temporarily changed for load test.")
    time.sleep(1)

    logger.info(f"\nLoading protocol from {protocol_filepath} to Channel 0...")
    if system.load_protocol(filepath=protocol_filepath, channel_id=0):
        logger.info("Protocol loaded to Channel 0.")
        loaded_params = system.get_parameters(0)
        logger.info(f"Channel 0 parameters after load: {loaded_params}")
    else:
        logger.error("Failed to load protocol to Channel 0.")
    time.sleep(1)

    # --- Demo 7: Get Status ---
    logger.info("\n--- Demo 7: Get Status ---")
    status_all = system.get_status()
    logger.info("\nCurrent system status:")
    for ch_status in status_all:
        logger.info(f"  Channel {ch_status['channel_id']}: Stimulating={ch_status['is_stimulating']}, "
              f"Amp={ch_status['parameters']['amplitude_mA']:.1f}mA, Freq={ch_status['parameters']['frequency_hz']}Hz")
    time.sleep(1)

    # --- Final Cleanup ---
    logger.info("\n--- Final Cleanup ---")
    logger.info("\nClosing stimulation module and stopping all remaining stimulation...")
    system.close()
    logger.info("Demonstration complete.")

if __name__ == "__main__":
    main()
