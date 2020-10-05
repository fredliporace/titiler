"""CBERS cloud demo endpoint."""

# Extra packages:
# tensorflow, keras

import os
from dataclasses import dataclass
from typing import Dict, Optional, Type
from urllib.parse import urlencode

import numpy as np
import pkg_resources
#from keras.models import load_model
import tflite_runtime.interpreter as tflite
from morecantile import TileMatrixSet
from rasterio.transform import from_bounds
from rio_tiler_crs import STACReader

from .. import utils
from ..dependencies import DefaultDependency
from ..models.dataset import Info, Metadata
from ..models.mapbox import TileJSON
from ..ressources.common import img_endpoint_params
from ..ressources.enums import ImageMimeTypes, ImageType  # fmt: off
from .factory import TMSTilerFactory

from fastapi import Depends, Path, Query

from starlette.requests import Request
from starlette.responses import Response
from starlette.templating import Jinja2Templates

# Keras inference
# NETWORK_FILE = "./keras-models/cloud_segmentation_20200923_1844.h5"
# MODEL = load_model(NETWORK_FILE)
# tflite inference

# assets: R, G, B, NIR bands
INSTRUMENT_PARAMS = {
    "LANDSAT_8": {
        "kwargs": {"assets": ["B4", "B3", "B2", "B5"],},
        "m2l_gains": [1.0, 1.0, 1.0, 1.0],
        "m2l_offsets": [0.0, 0.0, 0.0, 0.0],
    },
    "MUX": {
        "kwargs": {"assets": ["B7", "B6", "B5", "B8"]},
        # Parameters to obtain MUX from LS values
        # MUX = LS * gain + offset
        # Marco's e-mail, 23/9/2020
        # Order is R, G, B, NIR (MUX B7, B6, B5, B8)
        "m2l_gains": [0.0058, 0.0067, 0.0077, 0.0038],
        "m2l_offsets": [-27.7421, -31.9362, -38.3616, -13.3762],
    },
    "AWFI": {
        "kwargs": {"assets": ["B15", "B14", "B13", "B16"]},
        # Using the MUX parameters for now
        "m2l_gains": [0.0058, 0.0067, 0.0077, 0.0038],
        "m2l_offsets": [-27.7421, -31.9362, -38.3616, -13.3762],
    },
}

# M2L_GAINS = [0.0058, 0.0067, 0.0077, 0.0038]
# M2L_OFFSETS = [-27.7421, -31.9362, -38.3616, -13.3762]

template_dir = pkg_resources.resource_filename("titiler", "templates")
templates = Jinja2Templates(directory=template_dir)


@dataclass
class AssetsParams(DefaultDependency):
    """Assets, Band Indexes and Expression parameters."""

    assets: Optional[str] = Query(
        None,
        title="Asset indexes",
        description="comma (',') delimited asset names for R, G, B and NIR bands",
    )

    def __post_init__(self):
        """Post Init."""
        if self.assets is not None:
            self.kwargs["assets"] = self.assets.split(",")


@dataclass
class CBERSCloudTiler(TMSTilerFactory):
    """Custom Tiler for CBERS clouds from STAC."""

    reader: Type[STACReader] = STACReader

    layer_dependency: Type[DefaultDependency] = AssetsParams

    # Overwrite _tile method to compute cloud cover on the fly
    def tile(self):  # noqa: C901
        """Register /tiles endpoints."""
        tile_endpoint_params = img_endpoint_params.copy()

        @self.router.get(r"/tiles/{z}/{x}/{y}", **tile_endpoint_params)
        @self.router.get(r"/tiles/{z}/{x}/{y}.{format}", **tile_endpoint_params)
        @self.router.get(r"/tiles/{z}/{x}/{y}@{scale}x", **tile_endpoint_params)
        @self.router.get(
            r"/tiles/{z}/{x}/{y}@{scale}x.{format}", **tile_endpoint_params
        )
        @self.router.get(
            r"/tiles/{TileMatrixSetId}/{z}/{x}/{y}", **tile_endpoint_params
        )
        @self.router.get(
            r"/tiles/{TileMatrixSetId}/{z}/{x}/{y}.{format}", **tile_endpoint_params
        )
        @self.router.get(
            r"/tiles/{TileMatrixSetId}/{z}/{x}/{y}@{scale}x", **tile_endpoint_params
        )
        @self.router.get(
            r"/tiles/{TileMatrixSetId}/{z}/{x}/{y}@{scale}x.{format}",
            **tile_endpoint_params,
        )
        def tile(
            z: int = Path(..., ge=0, le=30, description="Mercator tiles's zoom level"),
            x: int = Path(..., description="Mercator tiles's column"),
            y: int = Path(..., description="Mercator tiles's row"),
            tms: TileMatrixSet = Depends(self.tms_dependency),
            scale: int = Query(
                1, gt=0, lt=4, description="Tile size scale. 1=256x256, 2=512x512..."
            ),
            format: ImageType = Query(
                None, description="Output image type. Default is auto."
            ),
            src_path=Depends(self.path_dependency),
            layer_params=Depends(self.layer_dependency),
            dataset_params=Depends(self.dataset_dependency),
            render_params=Depends(self.render_dependency),
            kwargs: Dict = Depends(self.additional_dependency),
        ):
            """Create map tile from a dataset."""
            timings = []
            headers: Dict[str, str] = {}

            tilesize = scale * 256

            # import pdb; pdb.set_trace()
            # src_path is the stac item json URL
            # layer_params is:
            # AssetsBidxExprParams(kwargs={'assets': ['B7', 'B6', 'B5']}, assets='B7,B6,B5', expression=None, bidx=None)
            # dataset_params is:
            # DatasetParams(kwargs={'resampling_method': 'nearest'}, nodata=None, unscale=None, resampling_method=<ResamplingNames.nearest: 'nearest'>)
            # kwargs is {}
            # STAC item with EO extension
            with utils.Timer() as t:
                with self.reader(
                    src_path.url, tms=tms, **self.reader_options
                ) as src_dst:
                    # import pdb; pdb.set_trace()
                    platform = src_dst.item["properties"]["platform"]
                    # If platform is CBERS-4 then the instrument is used as key
                    if platform == "CBERS-4":
                        platform = src_dst.item["properties"]["instruments"][0]
                    instrument_p = INSTRUMENT_PARAMS.get(platform)
                    assert instrument_p, f"Instrument {platform} not supported"
                    tile, mask = src_dst.tile(
                        x,
                        y,
                        z,
                        tilesize=tilesize,
                        # **layer_params.kwargs,
                        **instrument_p["kwargs"],
                        **dataset_params.kwargs,
                        **kwargs,
                    )
                    colormap = render_params.colormap or getattr(
                        src_dst, "colormap", None
                    )

            timings.append(("Read", t.elapsed))

            if not format:
                format = ImageType.jpg if mask.all() else ImageType.png

            with utils.Timer() as t:
                tile = utils.postprocess(
                    tile,
                    mask,
                    rescale=render_params.rescale,
                    color_formula=render_params.color_formula,
                )
            timings.append(("Post-process", t.elapsed))

            # import pdb; pdb.set_trace()
            # Change axis organization to NN convention,
            # bands is last.
            rtile = np.moveaxis(tile, 0, -1)
            # Add first dimension (single batch)
            rtile = np.expand_dims(rtile, axis=0)
            # Create input for network and apply MUX scaling
            stile = np.empty((1, 256, 256, 4))
            for i in range(0, 4):
                stile[0, :, :, i] = (
                    rtile[0, :, :, i] - INSTRUMENT_PARAMS[platform]["m2l_offsets"][i]
                ) / INSTRUMENT_PARAMS[platform]["m2l_gains"][i]
            # Prediction (Keras)
            # pred = MODEL.predict(stile)
            # Prediction (tflite)
            # INTERPRETER = interpreter = tflite.Interpreter(model_path='./keras-models/cloud_segmentation_20200923_1844.tflite')
            # INTERPRETER.allocate_tensors()
            # INPUT_TENSOR_INDEX = INTERPRETER.get_input_details()[0]['index']
            # OUTPUT_TENSOR_INDEX = INTERPRETER.get_output_details()[0]['index']
            # self.interpreter.allocate_tensors()
            interpreter = tflite.Interpreter(model_path='./cloud_segmentation_20200923_1844.tflite')
            interpreter.allocate_tensors()
            input_tensor_index = interpreter.get_input_details()[0]['index']
            output_tensor_index = interpreter.get_output_details()[0]['index']
            interpreter.set_tensor(input_tensor_index,
                                   np.array(stile, dtype=np.float32))
            interpreter.invoke()
            pred = interpreter.get_tensor(output_tensor_index)
            # Cloud mask from prediction
            cloud_mask = np.argmax(pred, axis=-1)
            cloud_mask[cloud_mask == 1] = 255

            # import pdb; pdb.set_trace()
            bounds = tms.xy_bounds(x, y, z)
            dst_transform = from_bounds(*bounds, tilesize, tilesize)
            with utils.Timer() as t:
                content = utils.reformat(
                    # tile[0:3],
                    cloud_mask.astype("uint8"),
                    mask if render_params.return_mask else None,
                    format,
                    colormap=colormap,
                    transform=dst_transform,
                    crs=tms.crs,
                )
            timings.append(("Format", t.elapsed))

            if timings:
                headers["X-Server-Timings"] = "; ".join(
                    [
                        "{} - {:0.2f}".format(name, time * 1000)
                        for (name, time) in timings
                    ]
                )

            return Response(
                content, media_type=ImageMimeTypes[format.value].value, headers=headers,
            )

        @self.router.get(
            "/tilejson.json",
            response_model=TileJSON,
            responses={200: {"description": "Return a tilejson"}},
            response_model_exclude_none=True,
        )
        @self.router.get(
            "/{TileMatrixSetId}/tilejson.json",
            response_model=TileJSON,
            responses={200: {"description": "Return a tilejson"}},
            response_model_exclude_none=True,
        )
        def tilejson(
            request: Request,
            tms: TileMatrixSet = Depends(self.tms_dependency),
            src_path=Depends(self.path_dependency),
            tile_format: Optional[ImageType] = Query(
                None, description="Output image type. Default is auto."
            ),
            tile_scale: int = Query(
                1, gt=0, lt=4, description="Tile size scale. 1=256x256, 2=512x512..."
            ),
            minzoom: Optional[int] = Query(
                None, description="Overwrite default minzoom."
            ),
            maxzoom: Optional[int] = Query(
                None, description="Overwrite default maxzoom."
            ),
            layer_params=Depends(self.layer_dependency),  # noqa
            dataset_params=Depends(self.dataset_dependency),  # noqa
            render_params=Depends(self.render_dependency),  # noqa
            kwargs: Dict = Depends(self.additional_dependency),  # noqa
        ):
            """Return TileJSON document for a dataset."""
            route_params = {
                "z": "{z}",
                "x": "{x}",
                "y": "{y}",
                "scale": tile_scale,
                "TileMatrixSetId": tms.identifier,
            }
            if tile_format:
                route_params["format"] = tile_format.value
            tiles_url = self.url_for(request, "tile", **route_params)

            q = dict(request.query_params)
            q.pop("TileMatrixSetId", None)
            q.pop("tile_format", None)
            q.pop("tile_scale", None)
            q.pop("minzoom", None)
            q.pop("maxzoom", None)
            qs = urlencode(list(q.items()))
            tiles_url += f"?{qs}"
            # import pdb; pdb.set_trace()

            with self.reader(src_path.url, tms=tms, **self.reader_options) as src_dst:
                center = list(src_dst.center)
                if minzoom:
                    center[-1] = minzoom
                tjson = {
                    "bounds": src_dst.bounds,
                    "center": tuple(center),
                    "minzoom": minzoom if minzoom is not None else src_dst.minzoom,
                    "maxzoom": maxzoom if maxzoom is not None else src_dst.maxzoom,
                    "name": os.path.basename(src_path.url),
                    "tiles": [tiles_url],
                }

            return tjson


cbers_cloud = CBERSCloudTiler(router_prefix="cberscloud")

router = cbers_cloud.router
