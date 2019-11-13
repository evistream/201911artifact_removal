#!/usr/bin/python

import maxlab
import maxlab.system
import maxlab.chip
import maxlab.util
import maxlab.saving #計測
import random
import time
import numpy as np

class Experiment:
    # chip,date
    def __init__(self, stim_patterns, save_dir, config_file, activity_file,stim_file):
        self.save_dir = save_dir
        self.name_of_configuration = config_file
        self.file_name = activity_file
        self.npz_file_name = stim_file
        self.stim_power = 34  # 34[bit] == 200mV
        self.stim_patterns = np.array(stim_patterns)
        flatten_stimpat = []
        for pat in self.stim_patterns:
            for i in pat:
                flatten_stimpat.append(i)
        self.electrodes = np.unique(flatten_stimpat)
        self.stimulation_times_each = 50  # stimulation times for each electrodes
        # 20000*8 -> overflow -> %= 65536 -> 28928
        self.stim_interval = 20000 * 10  # steps for wait; 20000steps == 1s
        # wait between configuring electrodes and stimulation
        self.pre_stim_interval = 20000 * 2
        
        self.MAX_DELAY = 65546-99
        self.wait_time = self.stimulation_times_each * \
            len(self.stim_patterns) * (self.stim_interval +
                self.pre_stim_interval) / 20000 * 1.01  # [s]
        self.test_wait_time = len(self.stim_patterns) * (self.stim_interval +
                self.pre_stim_interval) / 20000 * 1.01  # [s]

        # making stim_index,stim_electrodes_index
        size = len(self.stim_patterns)
        stim_index = np.arange(size)
        stim_index = np.tile(stim_index, self.stimulation_times_each)
        np.random.shuffle(stim_index)
        self.stim_index = stim_index

    def init(self):
        maxlab.util.initialize()
        maxlab.send(maxlab.chip.Core().enable_stimulation_power(True))

        array = maxlab.chip.Array('online')
        array.load_config(self.name_of_configuration)

        for e in self.electrodes:
            array.connect_electrode_to_stimulation(e)

        stimulations = {}
        count = 0
        for e in self.electrodes:
            stimulations[e] = array.query_stimulation_at_electrode(e)
            if stimulations[e]:
                count += 1

        if count != len(self.electrodes):
            print("Error: some electrode cannot be stimulated")
            return
        else:
            print("patterns is available")

        # Download the prepared array configuration to the chip
        array.download()

        # Prepare commands to power up and power down the two stimulation units
        cmd_power_stim = {}
        cmd_power_down_stim = {}
        for e in self.electrodes:
            cmd_power_stim[e] = maxlab.chip.StimulationUnit(stimulations[e]).power_up(
                True).connect(True).set_voltage_mode().dac_source(0)
            cmd_power_down_stim[e] = maxlab.chip.StimulationUnit(stimulations[e]).power_up(False)

        # 3. Prepare two different sequences of pulse trains

        def append_stimulation(seq, stim_index):
            for e in self.stim_patterns[stim_index]:
                seq.append(cmd_power_stim[e])

            for i in range(self.pre_stim_interval // self.MAX_DELAY):
                seq.append( maxlab.system.DelaySamples(self.MAX_DELAY))
            seq.append( maxlab.system.DelaySamples(self.pre_stim_interval % self.MAX_DELAY))

            def append_stimulation_pulse(seq, amplitude):
                seq.append( maxlab.chip.DAC(0, 512-amplitude) )
                seq.append( maxlab.system.DelaySamples(4) ) # 4/20k * 2 == 5kHz pulse
                seq.append( maxlab.chip.DAC(0, 512+amplitude) )
                seq.append( maxlab.system.DelaySamples(4) )
                seq.append( maxlab.chip.DAC(0, 512) )
                return seq
            append_stimulation_pulse(seq, self.stim_power) 

            for i in range(self.stim_interval // self.MAX_DELAY):
                seq.append( maxlab.system.DelaySamples(self.MAX_DELAY))
            seq.append( maxlab.system.DelaySamples(self.stim_interval % self.MAX_DELAY))

            for e in self.stim_patterns[stim_index]:
                seq.append( cmd_power_down_stim[e] )

        sequences = maxlab.Sequence() 
        for i in self.stim_index:
            append_stimulation(sequences,i)
        self.sequences = sequences

        testsequences = maxlab.Sequence() 
        for i in range(len(self.stim_patterns)):
            append_stimulation(testsequences,i)
        self.testsequences = testsequences

    def exec(self):
        # 計測開始
        start_time = time.time()
        s = maxlab.saving.Saving()
        s.open_directory(self.save_dir)
        s.start(self.file_name) #file_name
        print("start time: " + time.ctime(start_time) )
        print("plz wait:"+str(self.wait_time)+"[s]")

        # 刺激開始
        print("start stimulation")
        self.sequences.send()

        time.sleep(self.wait_time)

        # 計測終了
        s.stop()
        end_time = time.time()
        elasped = end_time - start_time

        print('\007') #beep
        print("elasped time[m]:" + str(elasped/60))

        np.savez(self.save_dir + self.npz_file_name ,electrodes=self.electrodes,stim_index=self.stim_index)
        print("saved stimulation config data at "+self.save_dir + self.npz_file_name )

    def test(self):

        # 計測開始
        start_time = time.time()
        s = maxlab.saving.Saving()
        s.open_directory(self.save_dir)
        s.start('test'+self.file_name) #file_name
        print("start time: " + time.ctime(start_time) )
        print("plz wait:"+str(self.test_wait_time)+"[s]")

        # 刺激開始
        print("start test stimulation")
        self.testsequences.send()

        time.sleep(self.test_wait_time)

        # 計測終了
        s.stop()
        end_time = time.time()
        elasped = end_time - start_time

        print('\007') #beep
        print("elasped time[m]:" + str(elasped/60))

        np.savez(self.save_dir + 'test' + self.npz_file_name ,electrodes=self.electrodes,stim_index=self.stim_index)
        print("saved stimulation config data at "+self.save_dir + self.npz_file_name )


def main():
    chip = 3635
    date = "190905"
    save_dir = "/home/maxone/Documents/sakurayama_workspace/201908/C{}/{}/".format(chip,date)
    stim_electrodes_file = save_dir + "configs/stm.txt"
    config_file = save_dir + "configs/000.cfg"
    activity_file = "activity"
    stim_file = "stimulation_config.npz"

    num_of_stm_ele = 4
    selectrodes = []
    with open(stim_electrodes_file) as f:
        lines = f.readlines()
        for l in lines[:num_of_stm_ele]:
            elec_id = int(l.strip())
            selectrodes.append(elec_id)
    selectrodes = np.array(selectrodes)

    stim_patterns = [
        [0,1,],  # 0
        [2,3,],  # 1
        [0,2,],  # 2
        [1,3,],  # 3
        [0,3,],  # 4
        [1,2,],  # 5
        [0,],  # 6
        [1,],   # 7
        [0,1,2,],  # 8
        [0,1,3,],  # 9
        [0,1,2,3,]  # 10
    ]
    for i,pat in enumerate(stim_patterns):
        for j,index in enumerate(pat):
            stim_patterns[i][j]=selectrodes[index]

    experiment = Experiment(stim_patterns, save_dir, config_file, activity_file, stim_file)
    experiment.init()
    experiment.exec()
    # experiment.test()

if __name__ == '__main__':
    main()
