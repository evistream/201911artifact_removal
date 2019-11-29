import numpy as np
import pathlib
from cached_property import cached_property
from sklearn.decomposition import PCA
from scipy import interpolate

from maxone_code.data import Experiment
from maxone_code.fig.multiunit import FigMultiUnit
from maxone_code.fig.mea import FigMea
from maxone_code.util import distElecId
from maxone_code.spikedetect import BandPassFilter,HighPassFilter
import maxone_code.util as util


class CheckFilter(object):

    # 電極・チャネルは以下のidで表現される。それらのうち一つが決まれば、すべて一意に決まる。
    # elec_id : 電極を表すid
    # chan_id : チャンネルを表すid
    # xid : 解析用データ列(data_*)上での列数または行数を表すid

    def __init__(self, path):
        self.path = path
        self.exp = Experiment(path)

        self.BPF_MARGIN_FRAME = 300
        self.SATU_VAL = 1023

        self.drawer = FigMultiUnit()
        self.meadrawer = FigMea()

        bpfs = []
        bpfs.append(BandPassFilter(lowcut=250, highcut=3000, order=4))
        bpfs.append(BandPassFilter(lowcut=250, highcut=3000, order=4))
        bpfs.append(BandPassFilter(lowcut=250, highcut=3000, order=4))
        bpfs.append(BandPassFilter(lowcut=250, highcut=3000, order=4))
        bpfs.append(BandPassFilter(lowcut=250, highcut=3000, order=4))
        bpfs.append(HighPassFilter(lowcut=250,order=4))
        self.bpf = BandPassFilter(lowcut=250, highcut=3000, order=4)

        self.assay_frame_range = None

        forcus_channels = list(self.exp.rawfile.channel2elec_table.keys())
        # forcus_channelsをelec_id順に並び替える
        def sortKey(chan):
            elec = self.exp.rawfile.channel2elec(chan)
            return elec
        forcus_channels.sort(key=sortKey)
        self.forcus_channels = forcus_channels
        self.SIZE_CHANNEL = len(self.forcus_channels)
        self.SIZE_MAX_CHANNEL = util.SIZE_CHANNEL # Maxoneの最大計測数

        channel2xid_table = {}
        xid2channel_table = {}
        for i, chan in enumerate(self.forcus_channels):
            channel2xid_table[chan] = i
            elec = self.exp.rawfile.channel2elec(chan)
            xid2channel_table[i] = (chan, elec)
        self.channel2xid_table = channel2xid_table
        self.xid2channel_table = xid2channel_table

    def disp(self):
        self.disp_raw()
        self.disp_filtered()

    def filterProccessing(self, filter_obj, raw_data, isContainMargin=True):
        if isContainMargin:
            return filter_obj.filter(raw_data)
        else:
            MARGIN = self.BPF_MARGIN_FRAME
            return filter_obj.filter(raw_data)[:, :, MARGIN:-MARGIN]

    def loaddata(self, cache_path=None, frame_slice=None):

        if frame_slice is None:
            frame_slice = np.s_[20000:20000*3]

        MARGIN = self.BPF_MARGIN_FRAME

        is_cache_exist = cache_path is not None and pathlib.Path(cache_path).exists()
        if is_cache_exist :
            self.data_raw_m = np.load(cache_path)
        else:
            scan_slice = np.s_[frame_slice.start - MARGIN: frame_slice.stop - MARGIN]
            assert (scan_slice.start > 0)

            LENGTH = scan_slice.stop-scan_slice.start
            data_pre_butter = np.zeros((1, self.SIZE_CHANNEL, LENGTH))
            source = self.exp.rawfile.sig
            channels = list(self.channel2xid_table.keys())

            SIZE_SIG_X = source.shape[0]
            data = np.zeros((SIZE_SIG_X, LENGTH))
            s_slice = np.s_[:, scan_slice]
            d_slice = np.s_[:, :LENGTH]
            source.read_direct(data, source_sel=s_slice, dest_sel=d_slice)
            data_pre_butter[0] = data[channels]

            self.data_raw_m = data_pre_butter  # 接尾語'_m'はマージン付きの意味
            if cache_path is not None:
                self.cache_data_raw_m(cache_path)

        self.data_raw = self.data_raw_m[:, :, MARGIN:-MARGIN]

        data_post_butter = self.filterProccessing(self.bpf,self.data_raw_m)
        self.data_filtered_m = data_post_butter
        self.data_filtered = data_post_butter[:, :, MARGIN:-MARGIN]

    def cache_data_raw_m(self,path):
        np.save(path, self.data_raw_m)

    @cached_property
    def record_eles(self):
        record_eles = {}
        for name, value in self.xid2channel_table.items():
            chan, _ = value
            posi = self.exp.rawfile.channel2posi(chan)
            record_eles[name] = posi
        return record_eles

    @cached_property
    def mea_xid_table(self):
        onset = np.array(list(self.record_eles.values())).min(axis=0)
        offset = np.array(list(self.record_eles.values())).max(axis=0)
        subplot_size = offset - onset + [1, 1]
        subplot_table = np.zeros(subplot_size, dtype=int)
        subplot_table.fill(-1)
        for k, posi in self.record_eles.items():
            index = np.array(list(posi)) - onset
            index = tuple(index)
            subplot_table[index] = k
        return subplot_table

    def filter_data_with_chan(self, data, drange, xids):
        data = data.copy()
        if drange is None:
            if xids is None:
                return data
            else:
                if isinstance(xids, int):
                    xids = [xids]
                return data[:, xids]
        else:
            raise(Exception)


    def disp_data(self, data, isConcat=True, ylabel=None, draw_xaxis=False, title=None, trange=None, yrange=None,
                  drange=None, frange=None, legend=False, xids=None, path=None, isFrameScale=True):
        # data ::= (trial, channel, frame)
        data = data.copy()

        # channel範囲
        # xidで指定
        data = self.filter_data_with_chan(data, drange, xids)

        # trial範囲
        if trange is not None:
            if isinstance(trange, int):
                data=data[[trange]]
            else:
                data=data[trange]

        # frame範囲
        if frange is not None:
            data = data[:, :, frange]
        else:
            if self.assay_frame_range is not None:
                data = data[:, :, self.assay_frame_range]

        # trialで結合させるかどうか
        if isConcat is True:
            size_ch = data.shape[1]  # chan
            data = data.transpose(1, 0, 2).reshape(1, size_ch, -1)

        for i, trial in enumerate(data):
            self.drawer.draw(trial, ylabel=ylabel, draw_xaxis=draw_xaxis, title=title, yrange=yrange,
                             legend=legend, path=None, isFrameScale=isFrameScale)

    def disp_raw(self, **kwargs):
        kwargs.setdefault('ylabel','[bit]')
        kwargs.setdefault('title','sig data')
        self.disp_data(self.data_raw, **kwargs)

    def disp_filtered(self, **kwargs):
        kwargs.setdefault('ylabel','[bit]')
        kwargs.setdefault('title','post bandpass filter')
        kwargs.setdefault('draw_xaxis',True)
        self.disp_data(self.data_filtered, **kwargs)

    def disp_mea(self, **kwargs):
        self.meadrawer.draw_all(self.record_eles, {}, **kwargs)

    def disp_mea_local(self, **kwargs):
        kwargs.setdefault('range','auto')
        self.meadrawer.draw_local(self.record_eles, {}, **kwargs)

    def subplot_template(self, table, center, stride, subplot_size=None, parity=None):
        # table ::= [[int]]
        # center ::= [int, int]
        # stride ::= int
        # parity ::= 0 | 1 | None. if 1, the new table include stim_elec
        table=table.copy()
        offset = center % stride
        new_center = center // stride
        new_table = table[offset[0]::stride,offset[1]::stride]

        if parity is not None:
            center_parity = np.sum(new_center) % 2
            if (center_parity + parity) % 2 == 0:
                new_table[::2,::2] = -1
                new_table[1::2,1::2] = -1
            else:
                new_table[::2,1::2] = -1
                new_table[1::2,::2] = -1

        if subplot_size is not None:
            sp_offset_x = max(new_center[0] - subplot_size[0] // 2, 0)
            sp_offset_y = max(new_center[1] - subplot_size[1] // 2, 0)
            new_table = new_table[sp_offset_x:sp_offset_x + subplot_size[0] ,
                        sp_offset_y:sp_offset_y + subplot_size[1] ]
        return new_table

    def disp_data_on_mea(self, data, **kwargs):
        kwargs.setdefault('share_axis', False)
        kwargs.setdefault('show_spine', True)
        sp_pattern=kwargs.pop('sp_pattern', None)  # ::= None | int
        center=np.array([0,0])
        if sp_pattern is not None:
            if isinstance(sp_pattern,int):
                subplot_patterns = [
                    (4,(10,10),0),
                    (2,(9,9),0),
                    (1,(5,5),None),
                ]
                sp_pattern=subplot_patterns[sp_pattern]
            subplot_table = self.subplot_template(self.mea_xid_table, center,*sp_pattern)
        else:
            subplot_table = self.mea_xid_table
        self.drawer.draw_on_mea(data, subplot_table, **kwargs)

    def disp_data_on_mea_2(self, data1, data2,**kwargs):
        kwargs.setdefault('share_axis', False)
        kwargs.setdefault('show_spine', True)
        sp_pattern=kwargs.pop('sp_pattern', None)  # ::= None | int
        center=np.array([0,0])
        if sp_pattern is not None:
            if isinstance(sp_pattern,int):
                subplot_patterns = [  # (stride, subplot_size, parity)
                    (4,(10,10),0),
                    (2,(9,9),0),
                    (1,(5,5),None),
                ]
                sp_pattern=subplot_patterns[sp_pattern]
            subplot_table = self.subplot_template(self.mea_xid_table, center,*sp_pattern)
        else:
            subplot_table = self.mea_xid_table
        colors=kwargs.pop('colors', None)  # ::= None | int
        self.drawer.draw_on_mea_2(data1, data2, subplot_table, colors=colors, **kwargs)

    def disp_raw_on_mea(self, **kwargs):
        kwargs.setdefault('title','sig data')
        self.disp_data_on_mea(self.data_raw, **kwargs)

    def disp_filtered_on_mea(self, **kwargs):
        kwargs.setdefault('title','post bandpass filter')
        self.disp_data_on_mea(self.data_filtered, **kwargs)
