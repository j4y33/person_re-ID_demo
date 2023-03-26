### Stream Info
    "camera_id" : uuid    # camera
    "track_id"  : uuid    # track
    "cluster_id": uuid    # merged tracks id for same person

### Track start time/offset
    "start": uint64       # frame_id

        
### Track stop time/offset
    "end": uint64         # frame_id
        
### Decisions
```
    status: str   # employee, visitor, buyer
```

### Object trajectory


    "trajectory": [
        {
            "box_id": uint32             # optional, global box coordinate/id in GRID
            "frame_offset": uint64,      # ordered id of frame from start
            
            "bounding_box": [uint32, uint32, uint32, uint32],  # [left, top, width, height]
            "keypoints": []
            "confidence": float,

            "center": [uint32, uint32],  # [x, y]

            side: str                    # ['front', 'back']

            embedding: float32[]
            image: str                   # jpeg encoded (optional)
        }
    ]


### To think

```
{    
    "camera_id"   : uuid                             # camera
    "track_id"    : uuid
    "frame_offset": uint64, 
    "bounding_box": [uint32, uint32, uint32, uint32] # [left, top, width, height]
    "center"      : [uint32, uint32],                # [x, y]
    "confidence"  : float                            # 
    "side"        : str                              # ['front', 'back']
    "embedding"   : float32[]
    "image"       : str                              # jpeg encoded (optional)
}
```
