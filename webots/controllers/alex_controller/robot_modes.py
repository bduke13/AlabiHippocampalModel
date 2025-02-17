"""
robot_modes.py

Each mode sets learning flags, then calls driver._movement_func(driver).
We remove dictionary lookups inside the mode function to reduce overhead.
"""

def mode_learn_ojas(driver):
    if driver.pcn:
        driver.pcn.enable_ojas = True
        driver.pcn.enable_stdp = False
    # no dictionary lookup here, just call the cached movement function
    driver._movement_func(driver)

def mode_learn_hebb(driver):
    if driver.pcn:
        driver.pcn.enable_ojas = True
        driver.pcn.enable_stdp = True
    driver._movement_func(driver)

def mode_dmtp(driver):
    if driver.pcn:
        driver.pcn.enable_ojas = True
        driver.pcn.enable_stdp = True
    driver._movement_func(driver)

def mode_exploit(driver):
    if driver.pcn:
        driver.pcn.enable_ojas = False
        driver.pcn.enable_stdp = False
    driver._movement_func(driver)

def mode_plotting(driver):
    if driver.pcn:
        driver.pcn.enable_ojas = False
        driver.pcn.enable_stdp = False
    driver._movement_func(driver)

def mode_manual_control(driver):
    if driver.pcn:
        driver.pcn.enable_ojas = False
        driver.pcn.enable_stdp = False
    driver._movement_func(driver)

def mode_recording(driver):
    if driver.pcn:
        driver.pcn.enable_ojas = False
        driver.pcn.enable_stdp = False
    driver._movement_func(driver)

def mode_learn_all(driver):
    frac = driver.step_count / float(driver.num_steps)
    if driver.pcn:
        if frac < 0.33:
            driver.pcn.enable_ojas = True
            driver.pcn.enable_stdp = False
        elif frac < 0.66:
            driver.pcn.enable_ojas = True
            driver.pcn.enable_stdp = True
        else:
            driver.pcn.enable_ojas = True
            driver.pcn.enable_stdp = True
    driver._movement_func(driver)

# We'll keep a dictionary but we'll only do this lookup once in driver initialization
MODE_FUNCTIONS = {
    "LEARN_OJAS": mode_learn_ojas,
    "LEARN_HEBB": mode_learn_hebb,
    "DMTP": mode_dmtp,
    "EXPLOIT": mode_exploit,
    "PLOTTING": mode_plotting,
    "MANUAL_CONTROL": mode_manual_control,
    "RECORDING": mode_recording,
    "LEARN_ALL": mode_learn_all
}

def get_mode_function(robot_mode):
    return MODE_FUNCTIONS.get(robot_mode.name, mode_plotting)
