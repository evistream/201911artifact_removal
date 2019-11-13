#!/usr/bin/python

import maxlab
import maxlab.system
import maxlab.chip
import maxlab.util
import maxlab.saving
import time
import datetime
import numpy as np

from maxone_code.util import RECORD_GAIN,SAMPLING_FREQ
from maxone_code.data import CfgFile

def scan(config_path,output_path,isOnlySpike=False):
    STIM_AMP = 17 # 100mV?
    STIM_LOOP = 3
    STIM_INTERVAL = 20000
    WAIT_INTERVAL = 20000
    WAIT_BUFFER = 20000*10
    PRE_EXP_WAIT = 4  # [s]

    # 0. Initialize system into a defined state
    maxlab.util.initialize()
    #maxlab.send(maxlab.chip.Core().enable_stimulation_power(False))
    maxlab.send(maxlab.chip.Amplifier().set_gain(RECORD_GAIN))

    # 1. Load configuration
    array = maxlab.chip.Array('recording')
    array.reset()
    array.clear_selected_electrodes()
    array.load_config(config_path)

    # Download the prepared array configuration to the chip
    array.download()

    # offset 必要かどうか　FIXME
    print("offset")
    maxlab.util.offset()
    time.sleep(4)

    # 計測電極
    cfg_file = CfgFile(config_path)
    chan_elec_table = cfg_file.chan_elec_table
    elec_ids = chan_elec_table[chan_elec_table>=0]
    chan_ids = np.where(chan_elec_table>=0)[0]

    # 刺激シーケンスの作成
    stimulations = {}
    for chan_id in chan_ids:
        elec_id = chan_elec_table[chan_id]
        array.connect_electrode_to_stimulation(elec_id)
        stimulation = array.query_stimulation_at_electrode(elec_id)
        if not stimulation:
            print("Error: electrode: " + str(elec_id) + " cannot be stimulated")
        else:
            array.disconnect_amplifier_from_stimulation(chan_id)
            stimulations[elec_id]=stimulation

    cmd_power_up_stim = {}
    cmd_power_down_stim = {}
    for elec_id,stim in stimulations.items():
        cmd_power_up_stim[elec_id] = maxlab.chip.StimulationUnit(stim).power_up(True).connect(True).set_voltage_mode().dac_source(0)
        cmd_power_down_stim[elec_id] = maxlab.chip.StimulationUnit(stim).power_up(False)

    def append_stimulation_pulse(seq, amplitude):
        seq.append(maxlab.chip.DAC(0, 512 - amplitude))
        seq.append(maxlab.system.DelaySamples(4))
        seq.append(maxlab.chip.DAC(0, 512 + amplitude))
        seq.append(maxlab.system.DelaySamples(4))
        seq.append(maxlab.chip.DAC(0, 512))
        return seq

    sequence = maxlab.Sequence()
    for elec_id,stim in stimulations.items():
        sequence.append(maxlab.system.DelaySamples(WAIT_INTERVAL))
        sequence.append(cmd_power_up_stim[elec_id])
        for i in range(STIM_LOOP):
            append_stimulation_pulse(sequence, STIM_AMP)
            sequence.append(maxlab.system.DelaySamples(STIM_INTERVAL))
        sequence.append(cmd_power_down_stim[elec_id])

    wait_length = ((8+STIM_INTERVAL)*STIM_LOOP + WAIT_INTERVAL)*len(stimulations) + WAIT_BUFFER

    # 計測開始
    length = wait_length // SAMPLING_FREQ + PRE_EXP_WAIT
    start_time = datetime.datetime.now()
    print("start recording [{}]".format(start_time.ctime()))
    print("plz wait: {0[0]}m {0[1]}s".format(divmod(length,60)))
    s = maxlab.saving.Saving()
    if isOnlySpike:
        s.start_spikes_only(output_path)
    else:
        s.start(output_path)

    time.sleep(PRE_EXP_WAIT)

    sequence.send()

    time.sleep(length)

    # 計測終了
    s.stop()
    end_time = datetime.datetime.now()
    elasped = end_time - start_time
    print('\007')  # beep
    print("end recording [{}]".format(end_time.ctime()))
    print("elasped time[min]: {0[0]}m {0[1]}s".format(divmod(elasped.seconds,60)))


