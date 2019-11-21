# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import base64


import numpy as np


from zarr.compat import PY2, Mapping
from zarr.errors import MetadataError
from zarr.util import json_dumps, json_loads


ZARR_FORMAT = 2


def parse_metadata(s):

    # Here we allow that a store may return an already-parsed metadata object,
    # or a string of JSON that we will parse here. We allow for an already-parsed
    # object to accommodate a consolidated metadata store, where all the metadata for
    # all groups and arrays will already have been parsed from JSON.

    if isinstance(s, Mapping):
        # assume metadata has already been parsed into a mapping object
        meta = s

    else:
        # assume metadata needs to be parsed as JSON
        meta = json_loads(s)

    return meta


def decode_array_metadata(s):
    meta = parse_metadata(s)

    # check metadata format
    zarr_format = meta.get('zarr_format', None)
    if zarr_format != ZARR_FORMAT:
        raise MetadataError('unsupported zarr format: %s' % zarr_format)

    # extract array metadata fields
    try:
        dtype = decode_dtype(meta['dtype'])
        compressor = meta['compressor']
        filters = meta['filters']
        fill_value = decode_fill_value(meta['fill_value'], dtype)
        meta = dict(
            zarr_format=meta['zarr_format'],
            shape=tuple(meta['shape']),
            chunks=tuple(meta['chunks']),
            dtype=dtype,
            compressor=compressor,
            fill_value=fill_value,
            order=meta['order'],
            filters=filters,
        )
    except Exception as e:
        raise MetadataError('error decoding metadata: %s' % e)
    else:
        return meta


def encode_array_metadata(meta):
    dtype = meta['dtype']
    sdshape = ()
    if dtype.subdtype is not None:
        dtype, sdshape = dtype.subdtype
    encoded_dtype, varlen = encode_dtype(dtype)

    filters = meta['filters'] or []
    if varlen:
        # from numcodecs import MsgPack
        filters.append(MsgPack().get_config())
    if not filters:
        filters = None

    meta = dict(
        zarr_format=ZARR_FORMAT,
        shape=meta['shape'] + sdshape,
        chunks=meta['chunks'],
        dtype=encoded_dtype,
        compressor=meta['compressor'],
        fill_value=encode_fill_value(meta['fill_value'], dtype),
        order=meta['order'],
        filters=filters,
    )
    return json_dumps(meta)


def encode_subfield_type(field):
    if type(field) == tuple and len(field) == 2:
        name, t = field
        from zarr.compat import text_type, binary_type
        # Attempt to recover h5py-string subfields, which are encoded like: ('|O', {'vlen': str})
        if type(t) == tuple and len(t) == 2:
            obj, metadata = t
            if obj == '|O' and \
                    (metadata == {'vlen': text_type} or
                     metadata == {'vlen': binary_type}):
                return name, 'O'  # metadata['vlen']

        return name, t

    return field


def encode_dtype(d):
    if d.fields is None:
        return d.str, False
    else:
        fields = []
        varlen = False
        for field in d.descr:
            if type(field) == tuple and len(field) == 2:
                name, t = field
                from zarr.compat import text_type, binary_type
                # Attempt to recover h5py-string subfields, which are encoded like: ('|O', {'vlen': str})
                if type(t) == tuple and len(t) == 2:
                    obj, metadata = t
                    if obj == '|O' and \
                            (metadata == {'vlen': text_type} or
                             metadata == {'vlen': binary_type}):
                        fields.append((name, 'O'))   # metadata['vlen']
                        varlen = True
                else:
                    fields.append(field)
            else:
                fields.append(field)
        return fields, varlen


def _decode_dtype_descr(d):
    # need to convert list of lists to list of tuples
    print('decode dtype: %s (type %s)' % (d, type(d)))
    if isinstance(d, list):
        # recurse to handle nested structures
        if PY2:  # pragma: py3 no cover
            # under PY2 numpy rejects unicode field names
            d = [(k[0].encode("ascii"), _decode_dtype_descr(k[1])) + tuple(k[2:]) for k in d]
        else:  # pragma: py2 no cover
            d = [(k[0], _decode_dtype_descr(k[1])) + tuple(k[2:]) for k in d]
    elif isinstance(d, bytes):
        d = bytes.decode()
    return d


class DecodeDtypeException(Exception):
    pass


def decode_dtype(d):
    try:
        prev = d
        d = _decode_dtype_descr(d)
        print('decode_dtype: %s -> %s -> %s' % (prev, d, np.dtype(d)))
        return np.dtype(d)
    except KeyError:
        raise DecodeDtypeException("Failed to decode dtype: %s" % d)


def decode_group_metadata(s):
    meta = parse_metadata(s)

    # check metadata format version
    zarr_format = meta.get('zarr_format', None)
    if zarr_format != ZARR_FORMAT:
        raise MetadataError('unsupported zarr format: %s' % zarr_format)

    meta = dict(zarr_format=zarr_format)
    return meta


# N.B., keep `meta` parameter as a placeholder for future
# noinspection PyUnusedLocal
def encode_group_metadata(meta=None):
    meta = dict(
        zarr_format=ZARR_FORMAT,
    )
    return json_dumps(meta)


FLOAT_FILLS = {
    'NaN': np.nan,
    'Infinity': np.PINF,
    '-Infinity': np.NINF
}


def decode_fill_value(v, dtype):
    print('decode_fill_value: %s %s' % (v, dtype))
    # early out
    if v is None:
        return v
    if dtype.kind is 'f':
        if v == 'NaN':
            return np.nan
        elif v == 'Infinity':
            return np.PINF
        elif v == '-Infinity':
            return np.NINF
        else:
            return np.array(v, dtype=dtype)[()]
    elif dtype.kind is 'c':
        v = (decode_fill_value(v[0], dtype.type().real.dtype),
             decode_fill_value(v[1], dtype.type().imag.dtype))
        v = v[0] + 1j * v[1]
        return np.array(v, dtype=dtype)[()]
    elif dtype.kind == 'S':
        # noinspection PyBroadException
        try:
            v = base64.standard_b64decode(v)
        except Exception:
            # be lenient, allow for other values that may have been used before base64
            # encoding and may work as fill values, e.g., the number 0
            pass
        v = np.array(v, dtype=dtype)[()]
        return v
    elif dtype.kind == 'V':
        d = base64.standard_b64decode(v)
        if dtype.hasobject and dtype.fields:
            if all(( b == 0 for b in d )):
                # fill_value = []
                # for name, (field, alignment) in dtype.fields.items():
                #     start = alignment
                #     end = alignment + field.itemsize
                #     if field.kind != 'O':
                #         v = d[start:end]
                #         v = np.frombuffer(v, field, count=1)[0]
                #     else:
                #         v = None
                #     print('bytes [%d:%d): %s' % (start, end, v))
                #     field_fill_value = decode_fill_value(v, field)
                #     fill_value.append(field_fill_value)
                # print('wrap fill_value %s for dtype %s' % (fill_value, dtype))
                # return np.array(tuple(fill_value), dtype=dtype)
                return np.empty((), dtype=dtype)
            else:
                raise Exception('Non-zero fill_value not supported for structured dtype containing object')
        else:
            # print('decoded %s as %s' % (v, d))
            a = np.array(d, dtype=dtype.str).view(dtype)[()]
            return a
    elif dtype.kind == 'U':
        # leave as-is
        return v
    else:
        return np.array(v, dtype=dtype)[()]


def encode_fill_value(v, dtype):
    # early out
    if v is None:
        return v
    if dtype.kind == 'f':
        if np.isnan(v):
            return 'NaN'
        elif np.isposinf(v):
            return 'Infinity'
        elif np.isneginf(v):
            return '-Infinity'
        else:
            return float(v)
    elif dtype.kind in 'ui':
        return int(v)
    elif dtype.kind == 'b':
        return bool(v)
    elif dtype.kind in 'c':
        v = (encode_fill_value(v.real, dtype.type().real.dtype),
             encode_fill_value(v.imag, dtype.type().imag.dtype))
        return v
    elif dtype.kind in 'SV':
        e = base64.standard_b64encode(v)
        print('Encoded %s (%s) as %s' % (v, type(v), e))
        if not PY2:  # pragma: py2 no cover
            e = str(e, 'ascii')
        return e
    elif dtype.kind == 'U':
        return v
    elif dtype.kind in 'mM':
        return int(v.view('i8'))
    else:
        return v
