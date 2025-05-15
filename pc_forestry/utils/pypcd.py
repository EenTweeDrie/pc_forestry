import numpy as np
import copy
import warnings
import struct
import lzf


def write_header(metadata, rename_padding=False):
    """ Given metadata as dictionary, return a string header.
    """
    template = """\
    VERSION {version}
    FIELDS {fields}
    SIZE {size}
    TYPE {type}
    COUNT {count}
    WIDTH {width}
    HEIGHT {height}
    VIEWPOINT {viewpoint}
    POINTS {points}
    DATA {data}
    """
    str_metadata = metadata.copy()

    if not rename_padding:
        str_metadata['fields'] = ' '.join(metadata['fields'])
    else:
        new_fields = []
        for f in metadata['fields']:
            if f == '_':
                new_fields.append('padding')
            else:
                new_fields.append(f)
        str_metadata['fields'] = ' '.join(new_fields)
    str_metadata['size'] = ' '.join(map(str, metadata['size']))
    str_metadata['type'] = ' '.join(metadata['type'])
    str_metadata['count'] = ' '.join(map(str, metadata['count']))
    str_metadata['width'] = str(metadata['width'])
    str_metadata['height'] = str(metadata['height'])
    str_metadata['viewpoint'] = ' '.join(map(str, metadata['viewpoint']))
    str_metadata['points'] = str(metadata['points'])
    tmpl = template.format(**str_metadata)
    return tmpl


def decode_rgb_from_pcl(rgb):
    """ Decode the bit-packed RGBs used by PCL.

    :param rgb: An Nx1 array.
    :rtype: Nx3 uint8 array with one column per color.
    """

    rgb = rgb.copy()
    rgb.dtype = np.uint32
    r = np.asarray((rgb >> 16) & 255, dtype=np.uint8)
    g = np.asarray((rgb >> 8) & 255, dtype=np.uint8)
    b = np.asarray(rgb & 255, dtype=np.uint8)
    rgb_arr = np.zeros((rgb.size, 3), dtype=np.uint8)
    rgb_arr[:, 0] = r
    rgb_arr[:, 1] = g
    rgb_arr[:, 2] = b
    return rgb_arr


def build_ascii_fmtstr(pc):
    """ Make a format string for printing to ascii.

    Note %.8f is minimum for rgb.
    """
    fmtstr = []
    for t, cnt in zip(pc.type, pc.count):
        if t == 'F':
            fmtstr.extend(['%.10e']*cnt)
        elif t == 'I':
            fmtstr.extend(['%d']*cnt)
        elif t == 'U':
            fmtstr.extend(['%u']*cnt)
        else:
            raise ValueError("don't know about type %s" % t)
    return fmtstr


def point_cloud_to_fileobj(pc, fileobj, data_compression=None):
    """ Write pointcloud as .pcd to fileobj.
    If data_compression is not None it overrides pc.data.
    """
    metadata = pc.get_metadata()
    if data_compression is not None:
        data_compression = data_compression.lower()
        assert (data_compression in ('ascii', 'binary', 'binary_compressed'))
        metadata['data'] = data_compression

    header = write_header(metadata).encode('utf-8')
    fileobj.write(header)
    if metadata['data'].lower() == 'ascii':
        fmtstr = build_ascii_fmtstr(pc)
        if pc.pc_data.size == 1:
            np.savetxt(fileobj, pc.pc_data.reshape((1,)), fmt=fmtstr)
        else:
            np.savetxt(fileobj, pc.pc_data, fmt=fmtstr)

    elif metadata['data'].lower() == 'binary':
        fileobj.write(pc.pc_data.tostring())
    elif metadata['data'].lower() == 'binary_compressed':
        uncompressed_lst = []
        for fieldname in pc.pc_data.dtype.names:
            column = np.ascontiguousarray(pc.pc_data[fieldname]).tostring()
            uncompressed_lst.append(column)
        uncompressed = b''.join(uncompressed_lst)
        uncompressed_size = len(uncompressed)
        buf = lzf.compress(uncompressed)
        if buf is None:
            buf = uncompressed
            compressed_size = uncompressed_size
        else:
            compressed_size = len(buf)
        fmt = 'II'
        fileobj.write(struct.pack(fmt, compressed_size, uncompressed_size))
        fileobj.write(buf)
    else:
        raise ValueError('unknown DATA type')


class PointCloud(object):
    def __init__(self, metadata, pc_data):
        self.metadata_keys = metadata.keys()
        self.__dict__.update(metadata)
        self.pc_data = pc_data
        self.check_sanity()

    def get_metadata(self):
        """ returns copy of metadata """
        metadata = {}
        for k in self.metadata_keys:
            metadata[k] = copy.copy(getattr(self, k))
        return metadata

    def save(self, fname):
        self.save_pcd(fname, 'ascii')

    def save_pcd(self, fname, compression=None, **kwargs):
        if 'data_compression' in kwargs:
            warnings.warn('data_compression keyword is deprecated for'
                          ' compression')
            compression = kwargs['data_compression']
        with open(fname, 'wb') as f:
            point_cloud_to_fileobj(self, f, compression)
