from io import BufferedReader, BufferedWriter
from struct import unpack, pack
from warnings import warn
import math
from pathlib import Path

DEFAULT_MIDI_HEADER_SIZE = 14


def read_varlen(data):
    value = 0
    for chr in data:
        # shift last value up 7 bits
        value = value << 7
        # mask out the 8th bit
        # add new value
        value += chr & 0x7F
        # is the hi-bit set?
        if not (chr & 0x80):
            break
    return value


def write_varlen(value: int) -> bytes:
    chr1 = value & 0x7F
    value >>= 7
    if value:
        chr2 = (value & 0x7F) | 0x80
        value >>= 7
        if value:
            chr3 = (value & 0x7F) | 0x80
            value >>= 7
            if value:
                chr4 = (value & 0x7F) | 0x80
                res = bytes([chr4, chr3, chr2, chr1])
            else:
                res = bytes([chr3, chr2, chr1])
        else:
            res = bytes([chr2, chr1])
    else:
        res = bytes([chr1])
    return res


class EventMetaClass(type):
    def __init__(cls, name, bases, dict):
        if name not in [
            "AbstractEvent",
            "Event",
            "MetaEvent",
            "NoteEvent",
            "MetaEventWithText",
        ]:
            EventRegistry.register_event(cls, bases)


class AbstractEvent(metaclass=EventMetaClass):
    name = "Generic MIDI Event"
    length = 0
    statusmsg = 0x0
    tick = 0
    data = None

    def __init__(self, **kw):
        if type(self.length) == int:
            defdata = [0] * self.length
        else:
            defdata = []
        self.tick = 0
        self.data = defdata
        for key in kw:
            setattr(self, key, kw[key])

    def __cmp__(self, other):
        if self.tick < other.tick:
            return -1
        elif self.tick > other.tick:
            return 1
        return self.data == other.data

    def __baserepr__(self, keys=None):
        if keys is None:
            keys = []

        keys = ["tick"] + keys + ["data"]
        body = []
        for key in keys:
            val = getattr(self, key)
            body.append(f"{key}={val}")
        body = ", ".join(body)
        return f"midi.{self.__class__.__name__}({body})"

    def __repr__(self):
        return self.__baserepr__()


class Event(AbstractEvent):
    name = "Event"
    channel = 0

    def __init__(self, **kw):
        if "channel" not in kw:
            kw = kw.copy()
            kw["channel"] = 0
        super(Event, self).__init__(**kw)

    def copy(self, **kw):
        _kw = {"channel": self.channel, "tick": self.tick, "data": self.data}
        _kw.update(kw)
        return self.__class__(**_kw)

    def __cmp__(self, other):
        if self.tick < other.tick:
            return -1
        elif self.tick > other.tick:
            return 1
        return 0

    def __repr__(self):
        return self.__baserepr__(["channel"])

    def is_event(cls, statusmsg):
        return cls.statusmsg == (statusmsg & 0xF0)

    is_event = classmethod(is_event)


"""
MetaEvent is a special subclass of Event that is not meant to
be used as a concrete class.  It defines a subset of Events known
as the Meta events.
"""


class MetaEvent(AbstractEvent):
    statusmsg = 0xFF
    metacommand = 0x0
    name = "Meta Event"

    def is_event(cls, statusmsg):
        return statusmsg == 0xFF

    is_event = classmethod(is_event)


class EventRegistry(object):
    Events = {}
    MetaEvents = {}

    def register_event(cls, event, bases):
        if (Event in bases) or (NoteEvent in bases):
            assert event.statusmsg not in cls.Events, (
                "Event %s already registered" % event.name
            )
            cls.Events[event.statusmsg] = event
        elif (MetaEvent in bases) or (MetaEventWithText in bases):
            if event.metacommand is not None:
                assert event.metacommand not in cls.MetaEvents, (
                    "Event %s already registered" % event.name
                )
                cls.MetaEvents[event.metacommand] = event
        else:
            raise ValueError("Unknown bases class in event type: " + event.name)

    register_event = classmethod(register_event)


"""
NoteEvent is a special subclass of Event that is not meant to
be used as a concrete class.  It defines the generalities of NoteOn
and NoteOff events.
"""


class NoteEvent(Event):
    length = 2

    @property
    def pitch(self):
        return self.data[0]

    @pitch.setter
    def pitch(self, pitch):
        self.data[0] = pitch

    @property
    def velocity(self):
        return self.data[1]

    @velocity.setter
    def velocity(self, velocity):
        self.data[1] = velocity


class NoteOnEvent(NoteEvent):
    statusmsg = 0x90
    name = "Note On"


class NoteOffEvent(NoteEvent):
    statusmsg = 0x80
    name = "Note Off"


class AfterTouchEvent(Event):
    statusmsg = 0xA0
    length = 2
    name = "After Touch"

    @property
    def pitch(self):
        return self.data[0]

    @pitch.setter
    def pitch(self, pitch):
        self.data[0] = pitch

    @property
    def value(self):
        return self.data[1]

    @value.setter
    def value(self, value):
        self.data[1] = value


class ControlChangeEvent(Event):
    statusmsg = 0xB0
    length = 2
    name = "Control Change"

    @property
    def control(self):
        return self.data[0]

    @control.setter
    def control(self, value):
        self.data[0] = value

    @property
    def value(self):
        return self.data[1]

    @value.setter
    def value(self, value):
        self.data[1] = value


class ProgramChangeEvent(Event):
    statusmsg = 0xC0
    length = 1
    name = "Program Change"

    @property
    def value(self):
        return self.data[0]

    @value.setter
    def value(self, value):
        self.data[0] = value


class ChannelAfterTouchEvent(Event):
    statusmsg = 0xD0
    length = 1
    name = "Channel After Touch"

    @property
    def value(self):
        return self.data[0]

    @value.setter
    def value(self, value):
        self.data[0] = value


class PitchWheelEvent(Event):
    statusmsg = 0xE0
    length = 2
    name = "Pitch Wheel"

    @property
    def pitch(self):
        return ((self.data[1] << 7) | self.data[0]) - 0x2000

    @pitch.setter
    def pitch(self, pitch):
        value = pitch + 0x2000
        self.data[0] = value & 0x7F
        self.data[1] = (value >> 7) & 0x7F


class SysexEvent(Event):
    statusmsg = 0xF0
    name = "SysEx"
    length = "varlen"

    def is_event(cls, statusmsg):
        return cls.statusmsg == statusmsg

    is_event = classmethod(is_event)


class SequenceNumberMetaEvent(MetaEvent):
    name = "Sequence Number"
    metacommand = 0x00
    length = 2


class MetaEventWithText(MetaEvent):
    def __init__(self, **kw):
        super(MetaEventWithText, self).__init__(**kw)
        if "text" not in kw:
            self.text = bytes(self.data)

    def __repr__(self):
        return self.__baserepr__(["text"])


class TextMetaEvent(MetaEventWithText):
    name = "Text"
    metacommand = 0x01
    length = "varlen"


class CopyrightMetaEvent(MetaEventWithText):
    name = "Copyright Notice"
    metacommand = 0x02
    length = "varlen"


class TrackNameEvent(MetaEventWithText):
    name = "Track Name"
    metacommand = 0x03
    length = "varlen"


class InstrumentNameEvent(MetaEventWithText):
    name = "Instrument Name"
    metacommand = 0x04
    length = "varlen"


class LyricsEvent(MetaEventWithText):
    name = "Lyrics"
    metacommand = 0x05
    length = "varlen"


class MarkerEvent(MetaEventWithText):
    name = "Marker"
    metacommand = 0x06
    length = "varlen"


class CuePointEvent(MetaEventWithText):
    name = "Cue Point"
    metacommand = 0x07
    length = "varlen"


class ProgramNameEvent(MetaEventWithText):
    name = "Program Name"
    metacommand = 0x08
    length = "varlen"


class UnknownMetaEvent(MetaEvent):
    name = "Unknown"
    # This class variable must be overriden by code calling the constructor,
    # which sets a local variable of the same name to shadow the class variable.
    metacommand = None

    def __init__(self, **kw):
        super(MetaEvent, self).__init__(**kw)
        self.metacommand = kw["metacommand"]

    def copy(self, **kw):
        kw["metacommand"] = self.metacommand
        return super(UnknownMetaEvent, self).copy(kw)


class ChannelPrefixEvent(MetaEvent):
    name = "Channel Prefix"
    metacommand = 0x20
    length = 1


class PortEvent(MetaEvent):
    name = "MIDI Port/Cable"
    metacommand = 0x21


class TrackLoopEvent(MetaEvent):
    name = "Track Loop"
    metacommand = 0x2E


class EndOfTrackEvent(MetaEvent):
    name = "End of Track"
    metacommand = 0x2F


class SetTempoEvent(MetaEvent):
    name = "Set Tempo"
    metacommand = 0x51
    length = 3

    @property
    def bpm(self):
        return float(6e7) / self.mpqn

    @bpm.setter
    def bpm(self, bpm):
        self.mpqn = int(float(6e7) / bpm)

    @property
    def mpqn(self):
        assert len(self.data) == 3
        vals = [self.data[x] << (16 - (8 * x)) for x in range(3)]
        return sum(vals)

    @mpqn.setter
    def mpqn(self, val):
        self.data = [(val >> (16 - (8 * x)) & 0xFF) for x in range(3)]


class SmpteOffsetEvent(MetaEvent):
    name = "SMPTE Offset"
    metacommand = 0x54


class TimeSignatureEvent(MetaEvent):
    name = "Time Signature"
    metacommand = 0x58
    length = 4

    @property
    def numerator(self):
        return self.data[0]

    @numerator.setter
    def numerator(self, val):
        self.data[0] = val

    @property
    def denominator(self):
        return 2 ** self.data[1]

    @denominator.setter
    def denominator(self, val):
        self.data[1] = int(math.log(val, 2))

    @property
    def metronome(self):
        return self.data[2]

    @metronome.setter
    def metronome(self, val):
        self.data[2] = val

    @property
    def thirtyseconds(self):
        return self.data[3]

    @thirtyseconds.setter
    def thirtyseconds(self, val):
        self.data[3] = val


class KeySignatureEvent(MetaEvent):
    name = "Key Signature"
    metacommand = 0x59
    length = 2

    @property
    def alternatives(self):
        d = self.data[0]
        return d - 256 if d > 127 else d

    @alternatives.setter
    def alternatives(self, val):
        self.data[0] = 256 + val if val < 0 else val

    @property
    def minor(self):
        return self.data[1]

    @minor.setter
    def minor(self, val):
        self.data[1] = val


class SequencerSpecificEvent(MetaEvent):
    name = "Sequencer Specific"
    metacommand = 0x7F


from pprint import pformat


class Pattern(list):
    def __init__(self, tracks=None, resolution=220, format=1, tick_relative=True):
        if tracks is None:
            tracks = []

        self.format = format
        self.resolution = resolution
        self.tick_relative = tick_relative
        super(Pattern, self).__init__(tracks)

    def __repr__(self):
        return f"midi.Pattern(format={self.format}, resolution={self.resolution}, tracks=\n{pformat(list(self))})"

    def make_ticks_abs(self):
        self.tick_relative = False
        for track in self:
            track.make_ticks_abs()

    def make_ticks_rel(self):
        self.tick_relative = True
        for track in self:
            track.make_ticks_rel()

    def __getitem__(self, item):
        if isinstance(item, slice):
            indices = item.indices(len(self))
            return Pattern(
                resolution=self.resolution,
                format=self.format,
                tracks=(super(Pattern, self).__getitem__(i) for i in range(*indices)),
            )
        else:
            return super(Pattern, self).__getitem__(item)

    def __getslice__(self, i, j):
        # The deprecated __getslice__ is still called when subclassing built-in types
        # for calls of the form List[i:j]
        return self.__getitem__(slice(i, j))


class Track(list):
    def __init__(self, events=None, tick_relative=True):
        if events is None:
            events = []
        self.tick_relative = tick_relative
        super(Track, self).__init__(events)

    def make_ticks_abs(self):
        if self.tick_relative:
            self.tick_relative = False
            running_tick = 0
            for event in self:
                event.tick += running_tick
                running_tick = event.tick

    def make_ticks_rel(self):
        if not self.tick_relative:
            self.tick_relative = True
            running_tick = 0
            for event in self:
                event.tick -= running_tick
                running_tick += event.tick

    def __getitem__(self, item):
        if isinstance(item, slice):
            indices = item.indices(len(self))
            return Track((super(Track, self).__getitem__(i) for i in range(*indices)))
        else:
            return super(Track, self).__getitem__(item)

    def __getslice__(self, i, j):
        # The deprecated __getslice__ is still called when subclassing built-in types
        # for calls of the form List[i:j]
        return self.__getitem__(slice(i, j))

    def __repr__(self):
        output = pformat(list(self)).replace("\n", "\n  ")
        return f"midi.Track(\n  {output})"


class FileReader:
    def read(self, midifile: BufferedReader) -> Pattern:
        pattern = self.parse_file_header(midifile)
        for track in pattern:
            self.parse_track(midifile, track)
        return pattern

    def parse_file_header(self, midifile: BufferedReader) -> Pattern:
        # First four bytes are MIDI header
        magic = midifile.read(4)
        if magic != b"MThd":
            raise TypeError("Bad header in MIDI file.")

        # next four bytes are header size
        # next two bytes specify the format version
        # next two bytes specify the number of tracks
        # next two bytes specify the resolution/PPQ/Parts Per Quarter
        # (in other words, how many ticks per quater note)
        data = unpack(">LHHH", midifile.read(10))
        hdrsz = data[0]
        format = data[1]
        tracks = [Track() for _ in range(data[2])]
        resolution = data[3]
        # XXX: the assumption is that any remaining bytes
        # in the header are padding
        if hdrsz > DEFAULT_MIDI_HEADER_SIZE:
            midifile.read(hdrsz - DEFAULT_MIDI_HEADER_SIZE)
        return Pattern(tracks=tracks, resolution=resolution, format=format)

    def parse_track_header(self, midifile: BufferedReader) -> int:
        # First four bytes are Track header
        magic = midifile.read(4)
        if magic != b"MTrk":
            raise TypeError("Bad track header in MIDI file: " + magic)
        # next four bytes are track size
        trksz = unpack(">L", midifile.read(4))[0]
        return trksz

    def parse_track(self, midifile: BufferedReader, track: Track) -> None:
        self.RunningStatus = None
        trksz = self.parse_track_header(midifile)
        trackdata = iter(midifile.read(trksz))
        while True:
            try:
                event = self.parse_midi_event(trackdata)
                track.append(event)
            except StopIteration:
                break

    def parse_midi_event(self, trackdata: bytes) -> Event:
        # first datum is varlen representing delta-time
        tick = read_varlen(trackdata)
        # next byte is status message
        stsmsg = next(trackdata)
        # is the event a MetaEvent?
        if MetaEvent.is_event(stsmsg):
            cmd = next(trackdata)
            if cmd not in EventRegistry.MetaEvents:
                warn(f"Unknown Meta MIDI Event: {cmd}", Warning)
                cls = UnknownMetaEvent
            else:
                cls = EventRegistry.MetaEvents[cmd]
            datalen = read_varlen(trackdata)
            data = [next(trackdata) for _ in range(datalen)]
            return cls(tick=tick, data=data, metacommand=cmd)
        # is this event a Sysex Event?
        elif SysexEvent.is_event(stsmsg):
            data = []
            for datum in trackdata:
                if datum == 0xF7:
                    break
                data.append(datum)
            return SysexEvent(tick=tick, data=data)
        # not a Meta MIDI event or a Sysex event, must be a general message
        else:
            key = stsmsg & 0xF0
            if key not in EventRegistry.Events:
                assert self.RunningStatus, "Bad byte value"
                data = []
                key = self.RunningStatus & 0xF0
                cls = EventRegistry.Events[key]
                channel = self.RunningStatus & 0x0F
                data.append(stsmsg)
                data += [next(trackdata) for _ in range(cls.length - 1)]
                return cls(tick=tick, channel=channel, data=data)
            else:
                self.RunningStatus = stsmsg
                cls = EventRegistry.Events[key]
                channel = self.RunningStatus & 0x0F
                data = [next(trackdata) for _ in range(cls.length)]
                return cls(tick=tick, channel=channel, data=data)


class FileWriter:
    def write(self, midifile: BufferedWriter, pattern: Pattern) -> None:
        self.write_file_header(midifile, pattern)
        for track in pattern:
            self.write_track(midifile, track)

    def write_file_header(self, midifile: BufferedWriter, pattern: Pattern) -> None:
        # First four bytes are MIDI header
        packdata = pack(">LHHH", 6, pattern.format, len(pattern), pattern.resolution)
        midifile.write(b"MThd" + packdata)

    def write_track(self, midifile: BufferedWriter, track: Track) -> None:
        buf = b""
        self.RunningStatus = None
        for event in track:
            buf += self.encode_midi_event(event)
        buf = self.encode_track_header(len(buf)) + buf
        midifile.write(buf)

    def encode_track_header(self, trklen: int) -> bytes:
        return b"MTrk" + pack(">L", trklen)

    def encode_midi_event(self, event: Event) -> bytes:
        ret = b""
        ret += write_varlen(event.tick)
        # is the event a MetaEvent?
        if isinstance(event, MetaEvent):
            ret += bytes([event.statusmsg, event.metacommand])
            ret += write_varlen(len(event.data))
            ret += bytes(event.data)
        # is this event a Sysex Event?
        elif isinstance(event, SysexEvent):
            ret += bytes([0xF0] + event.data + [0xF7])
        # not a Meta MIDI event or a Sysex event, must be a general message
        elif isinstance(event, Event):
            if (
                not self.RunningStatus
                or self.RunningStatus.statusmsg != event.statusmsg
                or self.RunningStatus.channel != event.channel
            ):
                self.RunningStatus = event
                ret += bytes([event.statusmsg | event.channel])
            ret += bytes(event.data)
        else:
            raise ValueError("Unknown MIDI Event: " + str(event))
        return ret


def read_midifile(path: Path) -> Pattern:
    reader = FileReader()
    with open(path, "rb") as midifile:
        return reader.read(midifile)


def write_midifile(path: Path, pattern: Pattern) -> None:
    writer = FileWriter()
    with open(path, "wb") as midifile:
        writer.write(midifile, pattern)
