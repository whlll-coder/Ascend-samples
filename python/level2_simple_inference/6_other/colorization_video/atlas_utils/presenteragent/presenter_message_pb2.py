# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: presenter_message.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='presenter_message.proto',
  package='ascend.presenter.proto',
  syntax='proto3',
  serialized_pb=_b('\n\x17presenter_message.proto\x12\x16\x61scend.presenter.proto\"l\n\x12OpenChannelRequest\x12\x14\n\x0c\x63hannel_name\x18\x01 \x01(\t\x12@\n\x0c\x63ontent_type\x18\x02 \x01(\x0e\x32*.ascend.presenter.proto.ChannelContentType\"n\n\x13OpenChannelResponse\x12@\n\nerror_code\x18\x01 \x01(\x0e\x32,.ascend.presenter.proto.OpenChannelErrorCode\x12\x15\n\rerror_message\x18\x02 \x01(\t\"\x12\n\x10HeartbeatMessage\"\"\n\nCoordinate\x12\t\n\x01x\x18\x01 \x01(\r\x12\t\n\x01y\x18\x02 \x01(\r\"\x94\x01\n\x0eRectangle_Attr\x12\x34\n\x08left_top\x18\x01 \x01(\x0b\x32\".ascend.presenter.proto.Coordinate\x12\x38\n\x0cright_bottom\x18\x02 \x01(\x0b\x32\".ascend.presenter.proto.Coordinate\x12\x12\n\nlabel_text\x18\x03 \x01(\t\"\xb7\x01\n\x13PresentImageRequest\x12\x33\n\x06\x66ormat\x18\x01 \x01(\x0e\x32#.ascend.presenter.proto.ImageFormat\x12\r\n\x05width\x18\x02 \x01(\r\x12\x0e\n\x06height\x18\x03 \x01(\r\x12\x0c\n\x04\x64\x61ta\x18\x04 \x01(\x0c\x12>\n\x0erectangle_list\x18\x05 \x03(\x0b\x32&.ascend.presenter.proto.Rectangle_Attr\"o\n\x14PresentImageResponse\x12@\n\nerror_code\x18\x01 \x01(\x0e\x32,.ascend.presenter.proto.PresentDataErrorCode\x12\x15\n\rerror_message\x18\x02 \x01(\t*\xa5\x01\n\x14OpenChannelErrorCode\x12\x19\n\x15kOpenChannelErrorNone\x10\x00\x12\"\n\x1ekOpenChannelErrorNoSuchChannel\x10\x01\x12)\n%kOpenChannelErrorChannelAlreadyOpened\x10\x02\x12#\n\x16kOpenChannelErrorOther\x10\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01*P\n\x12\x43hannelContentType\x12\x1c\n\x18kChannelContentTypeImage\x10\x00\x12\x1c\n\x18kChannelContentTypeVideo\x10\x01*#\n\x0bImageFormat\x12\x14\n\x10kImageFormatJpeg\x10\x00*\xa4\x01\n\x14PresentDataErrorCode\x12\x19\n\x15kPresentDataErrorNone\x10\x00\x12$\n kPresentDataErrorUnsupportedType\x10\x01\x12&\n\"kPresentDataErrorUnsupportedFormat\x10\x02\x12#\n\x16kPresentDataErrorOther\x10\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01\x62\x06proto3')
)

_OPENCHANNELERRORCODE = _descriptor.EnumDescriptor(
  name='OpenChannelErrorCode',
  full_name='ascend.presenter.proto.OpenChannelErrorCode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='kOpenChannelErrorNone', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='kOpenChannelErrorNoSuchChannel', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='kOpenChannelErrorChannelAlreadyOpened', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='kOpenChannelErrorOther', index=3, number=-1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=780,
  serialized_end=945,
)
_sym_db.RegisterEnumDescriptor(_OPENCHANNELERRORCODE)

OpenChannelErrorCode = enum_type_wrapper.EnumTypeWrapper(_OPENCHANNELERRORCODE)
_CHANNELCONTENTTYPE = _descriptor.EnumDescriptor(
  name='ChannelContentType',
  full_name='ascend.presenter.proto.ChannelContentType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='kChannelContentTypeImage', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='kChannelContentTypeVideo', index=1, number=1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=947,
  serialized_end=1027,
)
_sym_db.RegisterEnumDescriptor(_CHANNELCONTENTTYPE)

ChannelContentType = enum_type_wrapper.EnumTypeWrapper(_CHANNELCONTENTTYPE)
_IMAGEFORMAT = _descriptor.EnumDescriptor(
  name='ImageFormat',
  full_name='ascend.presenter.proto.ImageFormat',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='kImageFormatJpeg', index=0, number=0,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=1029,
  serialized_end=1064,
)
_sym_db.RegisterEnumDescriptor(_IMAGEFORMAT)

ImageFormat = enum_type_wrapper.EnumTypeWrapper(_IMAGEFORMAT)
_PRESENTDATAERRORCODE = _descriptor.EnumDescriptor(
  name='PresentDataErrorCode',
  full_name='ascend.presenter.proto.PresentDataErrorCode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='kPresentDataErrorNone', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='kPresentDataErrorUnsupportedType', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='kPresentDataErrorUnsupportedFormat', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='kPresentDataErrorOther', index=3, number=-1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=1067,
  serialized_end=1231,
)
_sym_db.RegisterEnumDescriptor(_PRESENTDATAERRORCODE)

PresentDataErrorCode = enum_type_wrapper.EnumTypeWrapper(_PRESENTDATAERRORCODE)
kOpenChannelErrorNone = 0
kOpenChannelErrorNoSuchChannel = 1
kOpenChannelErrorChannelAlreadyOpened = 2
kOpenChannelErrorOther = -1
kChannelContentTypeImage = 0
kChannelContentTypeVideo = 1
kImageFormatJpeg = 0
kPresentDataErrorNone = 0
kPresentDataErrorUnsupportedType = 1
kPresentDataErrorUnsupportedFormat = 2
kPresentDataErrorOther = -1



_OPENCHANNELREQUEST = _descriptor.Descriptor(
  name='OpenChannelRequest',
  full_name='ascend.presenter.proto.OpenChannelRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='channel_name', full_name='ascend.presenter.proto.OpenChannelRequest.channel_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='content_type', full_name='ascend.presenter.proto.OpenChannelRequest.content_type', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=51,
  serialized_end=159,
)


_OPENCHANNELRESPONSE = _descriptor.Descriptor(
  name='OpenChannelResponse',
  full_name='ascend.presenter.proto.OpenChannelResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='error_code', full_name='ascend.presenter.proto.OpenChannelResponse.error_code', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='error_message', full_name='ascend.presenter.proto.OpenChannelResponse.error_message', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=161,
  serialized_end=271,
)


_HEARTBEATMESSAGE = _descriptor.Descriptor(
  name='HeartbeatMessage',
  full_name='ascend.presenter.proto.HeartbeatMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=273,
  serialized_end=291,
)


_COORDINATE = _descriptor.Descriptor(
  name='Coordinate',
  full_name='ascend.presenter.proto.Coordinate',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='ascend.presenter.proto.Coordinate.x', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='ascend.presenter.proto.Coordinate.y', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=293,
  serialized_end=327,
)


_RECTANGLE_ATTR = _descriptor.Descriptor(
  name='Rectangle_Attr',
  full_name='ascend.presenter.proto.Rectangle_Attr',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='left_top', full_name='ascend.presenter.proto.Rectangle_Attr.left_top', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='right_bottom', full_name='ascend.presenter.proto.Rectangle_Attr.right_bottom', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='label_text', full_name='ascend.presenter.proto.Rectangle_Attr.label_text', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=330,
  serialized_end=478,
)


_PRESENTIMAGEREQUEST = _descriptor.Descriptor(
  name='PresentImageRequest',
  full_name='ascend.presenter.proto.PresentImageRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='format', full_name='ascend.presenter.proto.PresentImageRequest.format', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='width', full_name='ascend.presenter.proto.PresentImageRequest.width', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='ascend.presenter.proto.PresentImageRequest.height', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data', full_name='ascend.presenter.proto.PresentImageRequest.data', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rectangle_list', full_name='ascend.presenter.proto.PresentImageRequest.rectangle_list', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=481,
  serialized_end=664,
)


_PRESENTIMAGERESPONSE = _descriptor.Descriptor(
  name='PresentImageResponse',
  full_name='ascend.presenter.proto.PresentImageResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='error_code', full_name='ascend.presenter.proto.PresentImageResponse.error_code', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='error_message', full_name='ascend.presenter.proto.PresentImageResponse.error_message', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=666,
  serialized_end=777,
)

_OPENCHANNELREQUEST.fields_by_name['content_type'].enum_type = _CHANNELCONTENTTYPE
_OPENCHANNELRESPONSE.fields_by_name['error_code'].enum_type = _OPENCHANNELERRORCODE
_RECTANGLE_ATTR.fields_by_name['left_top'].message_type = _COORDINATE
_RECTANGLE_ATTR.fields_by_name['right_bottom'].message_type = _COORDINATE
_PRESENTIMAGEREQUEST.fields_by_name['format'].enum_type = _IMAGEFORMAT
_PRESENTIMAGEREQUEST.fields_by_name['rectangle_list'].message_type = _RECTANGLE_ATTR
_PRESENTIMAGERESPONSE.fields_by_name['error_code'].enum_type = _PRESENTDATAERRORCODE
DESCRIPTOR.message_types_by_name['OpenChannelRequest'] = _OPENCHANNELREQUEST
DESCRIPTOR.message_types_by_name['OpenChannelResponse'] = _OPENCHANNELRESPONSE
DESCRIPTOR.message_types_by_name['HeartbeatMessage'] = _HEARTBEATMESSAGE
DESCRIPTOR.message_types_by_name['Coordinate'] = _COORDINATE
DESCRIPTOR.message_types_by_name['Rectangle_Attr'] = _RECTANGLE_ATTR
DESCRIPTOR.message_types_by_name['PresentImageRequest'] = _PRESENTIMAGEREQUEST
DESCRIPTOR.message_types_by_name['PresentImageResponse'] = _PRESENTIMAGERESPONSE
DESCRIPTOR.enum_types_by_name['OpenChannelErrorCode'] = _OPENCHANNELERRORCODE
DESCRIPTOR.enum_types_by_name['ChannelContentType'] = _CHANNELCONTENTTYPE
DESCRIPTOR.enum_types_by_name['ImageFormat'] = _IMAGEFORMAT
DESCRIPTOR.enum_types_by_name['PresentDataErrorCode'] = _PRESENTDATAERRORCODE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

OpenChannelRequest = _reflection.GeneratedProtocolMessageType('OpenChannelRequest', (_message.Message,), dict(
  DESCRIPTOR = _OPENCHANNELREQUEST,
  __module__ = 'presenter_message_pb2'
  # @@protoc_insertion_point(class_scope:ascend.presenter.proto.OpenChannelRequest)
  ))
_sym_db.RegisterMessage(OpenChannelRequest)

OpenChannelResponse = _reflection.GeneratedProtocolMessageType('OpenChannelResponse', (_message.Message,), dict(
  DESCRIPTOR = _OPENCHANNELRESPONSE,
  __module__ = 'presenter_message_pb2'
  # @@protoc_insertion_point(class_scope:ascend.presenter.proto.OpenChannelResponse)
  ))
_sym_db.RegisterMessage(OpenChannelResponse)

HeartbeatMessage = _reflection.GeneratedProtocolMessageType('HeartbeatMessage', (_message.Message,), dict(
  DESCRIPTOR = _HEARTBEATMESSAGE,
  __module__ = 'presenter_message_pb2'
  # @@protoc_insertion_point(class_scope:ascend.presenter.proto.HeartbeatMessage)
  ))
_sym_db.RegisterMessage(HeartbeatMessage)

Coordinate = _reflection.GeneratedProtocolMessageType('Coordinate', (_message.Message,), dict(
  DESCRIPTOR = _COORDINATE,
  __module__ = 'presenter_message_pb2'
  # @@protoc_insertion_point(class_scope:ascend.presenter.proto.Coordinate)
  ))
_sym_db.RegisterMessage(Coordinate)

Rectangle_Attr = _reflection.GeneratedProtocolMessageType('Rectangle_Attr', (_message.Message,), dict(
  DESCRIPTOR = _RECTANGLE_ATTR,
  __module__ = 'presenter_message_pb2'
  # @@protoc_insertion_point(class_scope:ascend.presenter.proto.Rectangle_Attr)
  ))
_sym_db.RegisterMessage(Rectangle_Attr)

PresentImageRequest = _reflection.GeneratedProtocolMessageType('PresentImageRequest', (_message.Message,), dict(
  DESCRIPTOR = _PRESENTIMAGEREQUEST,
  __module__ = 'presenter_message_pb2'
  # @@protoc_insertion_point(class_scope:ascend.presenter.proto.PresentImageRequest)
  ))
_sym_db.RegisterMessage(PresentImageRequest)

PresentImageResponse = _reflection.GeneratedProtocolMessageType('PresentImageResponse', (_message.Message,), dict(
  DESCRIPTOR = _PRESENTIMAGERESPONSE,
  __module__ = 'presenter_message_pb2'
  # @@protoc_insertion_point(class_scope:ascend.presenter.proto.PresentImageResponse)
  ))
_sym_db.RegisterMessage(PresentImageResponse)


# @@protoc_insertion_point(module_scope)
