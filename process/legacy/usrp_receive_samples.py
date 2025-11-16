import time
import uhd
import datetime
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

class Radio():
    RX_CLEAR_COUNT = 1000
    LO_ADJ = 1e6

    def __init__(self) -> None:
        self.usrp = uhd.usrp.MultiUSRP()
        self.usrp.set_rx_antenna("RX2")
        stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
        stream_args.channels = [0]
        self.channel = 0
        self.rxstreamer = self.usrp.get_rx_stream(stream_args)

    def _flush_rxstreamer(self):
        # For collecting metadata from radio command (i.e., errors, etc.)
        metadata = uhd.types.RXMetadata()
        # Figure out the size of the receive buffer and make it
        buffer_samps = self.rxstreamer.get_max_num_samps()
        recv_buffer = np.empty((1, buffer_samps), dtype=np.complex64)
        # Loop several times and read samples to clear out gunk.
        rx_stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        rx_stream_cmd.num_samps = buffer_samps * self.RX_CLEAR_COUNT
        rx_stream_cmd.stream_now = True
        self.rxstreamer.issue_stream_cmd(rx_stream_cmd)
        for i in range(self.RX_CLEAR_COUNT):
            samps = self.rxstreamer.recv(recv_buffer, metadata)
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                print(metadata.strerror())

    def tune(self, freq, gain, rate, use_lo_offset=False, gpiomode=None):
        # Set GPIO pins if provided
        if gpiomode is not None:
            self.usrp.set_gpio_attr("FP0", "CTRL", 0)
            self.usrp.set_gpio_attr("FP0", "DDR", 0x10)
            self.usrp.set_gpio_attr("FP0", "OUT", gpiomode)

		# Set rate (if provided)
        if rate:
            self.currate = rate

            # self.usrp.set_master_clock_rate(rate*2)
            print("Setting tx/rx rate = %f" %rate)
            self.usrp.set_tx_rate(rate, self.channel)
            self.usrp.set_rx_rate(rate, self.channel)


        ## Set tx bandwidth (if provided)
        #if bw:
            #self.usrp.set_tx_bandwidth(bw, self.channel)

        # Set the USRP freq
        if use_lo_offset:
            # Push the LO offset outside of sampling freq range
            lo_off = self.currate/2 + self.LO_ADJ
            self.usrp.set_rx_freq(uhd.types.TuneRequest(freq, lo_off), self.channel)
            self.usrp.set_tx_freq(uhd.types.TuneRequest(freq, lo_off), self.channel)
        else:
            self.usrp.set_rx_freq(uhd.types.TuneRequest(freq), self.channel)
            self.usrp.set_tx_freq(uhd.types.TuneRequest(freq), self.channel)
        

        # Set the USRP gain
        self.usrp.set_rx_gain(gain, self.channel)
        self.usrp.set_tx_gain(gain, self.channel)

        # Flush rx stream
        self._flush_rxstreamer()
    
    def recv_samples(self, nsamps, start_time=None, rate = None):
        # Set the sampling rate if necessary
        if rate and rate != self.currate:
            self.usrp.set_rx_rate(rate, self.channel)
            self._flush_rxstreamer()

        # Create the array to hold the return samples.
        samples = np.empty((1, int(nsamps)), dtype=np.complex64)

        # For collecting metadata from radio command (i.e., errors, etc.)
        metadata = uhd.types.RXMetadata()

        # Figure out the size of the receive buffer and make it
        buffer_samps = self.rxstreamer.get_max_num_samps()
        recv_buffer = np.zeros((1, buffer_samps), dtype=np.complex64)

        # Set up the device to receive exactly `nsamps` samples.
        rx_stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        rx_stream_cmd.num_samps = nsamps
        rx_stream_cmd.stream_now = True

        # Do synchronization
        if start_time:
            sleep_time = start_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Set up rx stream 
        self.rxstreamer.issue_stream_cmd(rx_stream_cmd)

        # Loop until we get the number of samples requested.  Append each
        # batch received to the return array.
        recv_samps = 0
        t = 0 # For debugging
        while recv_samps < nsamps:
            samps = self.rxstreamer.recv(recv_buffer, metadata, 0.1)

            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                print(metadata.strerror())
            if samps:
                #print(t, 'Got samples ', time.time())
                real_samps = min(nsamps - recv_samps, samps)
                samples[:, recv_samps:recv_samps + real_samps] = \
                    recv_buffer[:, 0:real_samps]
                recv_samps += real_samps
            #else:
                #print(t, 'No samples ', time.time())
            #t += 1

        dt = datetime.datetime.now()

        # Done.  Return samples.
        return samples, dt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--frequency", help="Frequency to receive samples", default=3534e6, type=float)
    parser.add_argument("-g", "--gain", help="Rx gain in dB", default=70, type=float)
    parser.add_argument("-r", "--rate", help="Rx sample rate", default=220e3, type=float)
    parser.add_argument("-n", "--nsamps", help="Number of samples to receive", default=131072, type=int)
    parser.add_argument("-w", "--sample_wait", help="Time between samples", default=1, type=float)
    parser.add_argument("-o", "--output_dir", help="Data directory", default='data', type=str)
    parser.add_argument("-l", "--use_lo_offset", help="Use low offset", default=True, type=bool)

    return parser.parse_args()

def main():
    args = parse_args()
    radio = Radio()

    # tune radio
    radio.tune(args.frequency, args.gain, args.rate, use_lo_offset=args.use_lo_offset)

    start_time = time.time()
    folder_name = datetime.datetime.now().strftime("samples_%Y%m%d-%H%M%S")
    folder_name = os.path.join(args.output_dir, folder_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    print('Saving results in ', folder_name)
    while(True):
        start_time = start_time + args.sample_wait
        samps, dt = radio.recv_samples(args.nsamps, start_time)
        filename = dt.strftime("%Y%m%d-%H%M%S.%f-IQ")
        filename += "rate%f_nsamps%i.npy" % (args.rate, args.nsamps)
        np.save(os.path.join(folder_name, filename), samps)
        print(filename, end='\r')

if __name__ == "__main__":
    main()
