import numpy

from scipy.ndimage import binary_erosion

from splendor.frame_buffer import FrameBufferWrapper
from splendor.camera import orthographic_matrix
from splendor.image import save_image, save_depth
from splendor.masks import color_byte_to_index

from ltron.geometry.utils import unscale_transform

def make_collision_framebuffer(resolution):
    frame_buffer = FrameBufferWrapper(
        resolution[0],
        resolution[1],
        anti_alias=False,
    )
    
    return frame_buffer

class CollisionChecker:
    def __init__(
        self,
        scene,
        resolution=(64,64),
        max_intersection=4,
        return_colliding_instances=False
    ):
        self.scene = scene
        self.frame_buffer = make_collision_framebuffer(resolution)
        self.max_intersection = max_intersection
        self.return_colliding_instances=False
    
    def check_collision(
        self,
        target_instances,
        render_transform,
        scene_instances=None
    ):
        return check_collision(
            self.scene,
            target_instances,
            render_transform,
            frame_buffer=self.frame_buffer,
            max_intersection=self.max_intersection,
            return_colliding_instances=self.return_colliding_instances
        )
    
    def check_snap_collision(
        self,
        target_instances,
        snap,
        direction,
        *args,
        **kwargs,
    ):
        return check_snap_collision(
            self.scene,
            target_instances,
            snap,
            direction,
            *args,
            frame_buffer=self.frame_buffer,
            return_colliding_instances=return_colliding_instances,
            **kwargs,
        )

def build_collision_map(
    scene,
    target_instances=None,
    scene_instances=None,
    frame_buffer=None,
    resolution=(64,64),
    *args,
    **kwargs,
):
    
    if target_instances is None:
        target_instances = set(str(i) for i in scene.instances.keys())
    
    if scene_instances is None:
        scene_instances = set(str(i) for i in scene.instances.keys())
    else:
        scene_instances = set(
            str(scene_instance) for scene_instance in scene_instances)
    
    if frame_buffer is None:
        frame_buffer = make_collision_framebuffer(resolution)
    
    edges = scene.get_all_edges(unidirectional=False)
    collision_map = {}
    for instance in target_instances:
        instance = scene.instances[instance]
        instance_id = instance.instance_id
        instance_name = instance.instance_name
        collision_map[instance_id] = {}
        source_edges = edges[0] == instance_id
        snaps_to_check = edges[2, source_edges]
        snap_groups = {}
        for snap_id in snaps_to_check:
            snap = instance.get_snap(snap_id)
            #feature = (snap.polarity == '+', *tuple(snap.transform[:3,1]))
            axis = snap.transform[:3,1]
            for key in snap_groups:
                #if numpy.allclose(key, feature):
                if numpy.abs(key @ axis) < 0.0001:
                    snap_groups[key].append(snap_id)
                    break
            else:
                #snap_groups[feature] = [snap_id]
                snap_groups[axis] = [snap_id]
        
        #for snap_id in snaps_to_check:
        #for feature, snap_ids in snap_groups.items():
        for axis, snap_ids in snap_groups.items():
            snap_id = snap_ids[0]
            snap = instance.get_snap(snap_id)
            for direction in 'push', 'pull':
                for s_id in snap_ids:
                    collision_map[instance_id][s_id, direction] = set()
                current_scene_instances = scene_instances - set([instance_name])
                k = 0
                while current_scene_instances:
                    #if instance_id == 1:
                    #    dump_images = 'one_%i_%s_%i'%(snap_id, direction, k)
                    #else:
                    #    dump_images = None
                    k += 1
                    colliders = check_snap_collision(
                        scene,
                        [instance],
                        snap,
                        direction,
                        scene_instances=current_scene_instances,
                        return_colliding_instances=True,
                        frame_buffer=frame_buffer,
                        #dump_images=dump_images,
                        *args,
                        **kwargs,
                    )
                    if len(colliders):
                        colliders = set(str(i) for i in colliders)
                        for s_id in snap_ids:
                            collision_map[instance_id][s_id, direction] |= (
                                colliders)
                        current_scene_instances -= colliders
                    else:
                        break
    
    return collision_map

def check_snap_collision(
    scene,
    target_instances,
    snap,
    direction,
    *args,
    **kwargs,
):
    
    assert direction in ('push', 'pull')
    
    if snap.polarity == '+':
        if direction == 'push':
            sign = 1
        elif direction == 'pull':
            sign = -1
    elif snap.polarity == '-':
        if direction == 'push':
            sign = -1
        elif direction == 'pull':
            sign = 1
    
    direction_transform = numpy.array([
        [ 1, 0,    0, 0],
        [ 0, 0, sign, 0],
        [ 0, 1,    0, 0],
        [ 0, 0,    0, 1]
    ])
    
    render_transform = snap.transform @ direction_transform
    
    return check_collision(
        scene,
        target_instances,
        render_transform,
        *args,
        **kwargs,
    )

def check_collision(
    scene,
    target_instances,
    render_transform,
    scene_instances=None,
    resolution=(64,64),
    frame_buffer=None,
    max_intersection=4,
    erosion=1,
    required_clearance=24,
    tolerance_spacing=8,
    dump_images=None,
    return_colliding_instances=False,
):
    
    # setup ====================================================================
    # make sure the scene is renderable
    assert scene.renderable
    
    # get a list of the names of the target and scene instances
    target_instance_names = set(
        str(target_instance) for target_instance in target_instances)
    if scene_instances is None:
        scene_instance_names = set(
            scene.get_all_brick_instances()) - target_instance_names
    else:
        scene_instance_names = set(
            str(scene_instance) for scene_instance in scene_instances)
    
    # build a splendor frame buffer if a shared one was not specified ----------
    if frame_buffer is None:
        frame_buffer = make_collision_framebuffer(resolution)
    
    # store the camera info and which bricks are hidden ------------------------
    original_view_matrix = scene.get_view_matrix()
    original_projection = scene.get_projection()
    
    # render the scene depth map ===============================================
    # setup the camera ---------------------------------------------------------
    camera_transform = unscale_transform(render_transform)
    render_axis = camera_transform[:3,2]
    
    # compute the extents of the tarrget instance in camera space --------------
    local_target_vertices = []
    inv_camera_transform = numpy.linalg.inv(camera_transform)
    for target_instance in target_instances:
        vertices = target_instance.brick_type.bbox_vertices
        transform = inv_camera_transform @ target_instance.transform
        local_target_vertices.append(transform @ vertices)
    local_target_vertices = numpy.concatenate(local_target_vertices, axis=1)
    box_min = numpy.min(local_target_vertices, axis=1)
    box_max = numpy.max(local_target_vertices, axis=1)
    thickness = box_max[2] - box_min[2]
    camera_distance = thickness + required_clearance + 2 * tolerance_spacing
    near_clip = 1 * tolerance_spacing
    far_clip = thickness * 2 + required_clearance + 3 * tolerance_spacing
    
    camera_transform[:3,3] += render_axis * camera_distance
    scene.set_view_matrix(numpy.linalg.inv(camera_transform))
    orthographic_projection = orthographic_matrix(
            l = box_max[0],
            r = box_min[0],
            b = -box_max[1],
            t = -box_min[1],
            n = near_clip,
            f = far_clip)
    
    scene.set_projection(orthographic_projection)
    
    # render -------------------------------------------------------------------
    frame_buffer.enable()
    scene.viewport_scissor(0,0,frame_buffer.width, frame_buffer.height)
    scene.mask_render(instances=scene_instance_names, ignore_hidden=True)
    if dump_images or return_colliding_instances:
        scene_mask = frame_buffer.read_pixels()
    scene_depth_map = frame_buffer.read_pixels(
            read_depth=True, projection=orthographic_projection)
    
    # render the target depth map ==============================================
    # setup the camera ---------------------------------------------------------
    camera_transform = unscale_transform(render_transform)
    camera_transform[:3,3] -= render_axis * camera_distance
    axis_flip = numpy.array([
        [ 1, 0, 0, 0],
        [ 0, 1, 0, 0],
        [ 0, 0,-1, 0],
        [ 0, 0, 0, 1]
    ])
    camera_transform = numpy.dot(camera_transform, axis_flip)
    scene.set_view_matrix(numpy.linalg.inv(camera_transform))
    scene.set_projection(orthographic_projection)
    
    # render -------------------------------------------------------------------
    frame_buffer.enable()
    scene.mask_render(instances=target_instance_names, ignore_hidden=True)
    target_mask = frame_buffer.read_pixels()
    target_depth_map = frame_buffer.read_pixels(
            read_depth=True, projection=orthographic_projection)
    
    # restore the previous camera ==============================================
    scene.set_view_matrix(original_view_matrix)
    scene.set_projection(original_projection)
    
    # check collision ==========================================================
    valid_pixels = numpy.sum(target_mask != 0, axis=-1) != 0
    
    scene_depth_map -= camera_distance
    scene_depth_map *= -1.
    target_depth_map -= camera_distance
    offset = (scene_depth_map - target_depth_map).reshape(valid_pixels.shape)
    offset *= valid_pixels
    
    # dump images ==============================================================
    if dump_images is not None:
        save_image(scene_mask, './%s_scene_mask.png'%dump_images)
        save_image(target_mask, './%s_target_mask.png'%dump_images)
        save_depth(scene_depth_map, './%s_scene_depth.npy'%dump_images)
        save_depth(target_depth_map, './%s_target_depth.npy'%dump_images)
        
        collision_pixels = (offset > max_intersection).astype(numpy.uint8)
        collision_pixels = collision_pixels * 255
        save_image(collision_pixels, './%s_collision.png'%dump_images)
    
    if erosion or return_colliding_instances:
        collision = offset > max_intersection
        if erosion:
            collision = binary_erosion(collision, iterations=erosion)
    
    if return_colliding_instances:
        colliding_y, colliding_x = numpy.where(collision)
        colliding_colors = scene_mask[colliding_y, colliding_x]
        colliding_bricks = numpy.unique(color_byte_to_index(colliding_colors))
        return colliding_bricks
    
    else:
        if erosion:
            collision = numpy.any(collision)
        else:
            collision = numpy.max(offset) > max_intersection
        return collision


def check_collision_old(
        scene,
        target_instances,
        snap_transform,
        target_snap_polarity,
        resolution=(64,64),
        frame_buffer=None,
        max_intersection=4,
        dump_images=None,
    ):
    
    # setup ====================================================================
    # make sure the scene is renderable
    assert scene.renderable
    
    # get a list of the non-target instances
    target_names = set(
        str(target_instance) for target_instance in target_instances)
    non_target_names = set(scene.get_all_brick_instances()) - target_names
    
    # build a splendor frame buffer if a shared one was not specified ----------
    if frame_buffer is None:
        frame_buffer = FrameBufferWrapper(
                resolution[0],
                resolution[1],
                anti_alias=False)
    
    # store the camera info and which bricks are hidden ------------------------
    original_view_matrix = scene.get_view_matrix()
    original_projection = scene.get_projection()
    #hidden_instances = {i : scene.instance_hidden(i) for i in scene.instances}
    
    # compute the camera distance, clipping plane and the orthgraphic width ----
    #camera_distance = 500 # TMP
    #orthographic_width = 100 # TMP
    #orthographic_height = 100 # TMP
    #near_clip = 1 # TMP
    #far_clip = 2000 # TMP
    
    # render the scene depth map ===============================================
    # show everything except for the target instances --------------------------
    
    # setup the camera ---------------------------------------------------------
    camera_transform = snap_transform.copy()
    render_axis = snap_transform[:3,1]
    render_axis /= numpy.linalg.norm(render_axis)
    p_direction = -1
    n_direction = 1
    p_rotate = numpy.array([
            [1, 0, 0, 0],
            [0, 0,-1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]])
    n_rotate = numpy.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]])

    if target_snap_polarity == '+':
        scene_axis = render_axis * n_direction
        scene_rotate = n_rotate
        target_axis = render_axis * p_direction
        target_rotate = p_rotate
    elif target_snap_polarity == '-':
        scene_axis = render_axis * p_direction
        scene_rotate = p_rotate
        target_axis = render_axis * n_direction
        target_rotate = n_rotate
    else:
        raise NotImplementedError
    
    # compute the relevant extents ---------------------------------------------
    local_vertices = []
    inv_snap_transform = numpy.linalg.inv(snap_transform)
    # this could be done by transforming the bounding box corners
    # (bbox of transformed bbox)
    for target_instance in target_instances:
        vertices = target_instance.brick_type.vertices
        transform = inv_snap_transform @ target_instance.transform
        local_vertices.append(transform @ vertices)
    local_vertices = numpy.concatenate(local_vertices, axis=1)
    box_min = numpy.min(local_vertices, axis=1)
    box_max = numpy.max(local_vertices, axis=1)
    thickness = box_max[1] - box_min[1]
    camera_distance = thickness + 2
    near_clip = 1
    far_clip = 2 * thickness + 3
    
    camera_transform[:3,3] += scene_axis * camera_distance
    camera_transform = numpy.dot(camera_transform, scene_rotate)
    scene.set_view_matrix(numpy.linalg.inv(camera_transform))
    orthographic_projection = orthographic_matrix(
            #l = -orthographic_width,
            #r = orthographic_width,
            #b = -orthographic_height,
            #t = orthographic_height,
            l = box_max[0],
            r = box_min[0],
            b = -box_max[2],
            t = -box_min[2],
            n = near_clip,
            f = far_clip)
    scene.set_projection(orthographic_projection)
    
    # render -------------------------------------------------------------------
    frame_buffer.enable()
    scene.mask_render(instances=non_target_names)
    scene_mask = frame_buffer.read_pixels()
    scene_depth_map = frame_buffer.read_pixels(
            read_depth=True, projection=orthographic_projection)
    
    #scene.color_render()
    #scene_color = frame_buffer.read_pixels()
    
    # render the instance depth map ============================================
    # hide everything except for the target instances --------------------------
    #scene.hide_all_instances()
    #for instance in target_instances:
    #    scene.show_instance(instance)
    
    # setup the camera ---------------------------------------------------------
    camera_transform = snap_transform.copy()
    camera_transform[:3,3] += target_axis * camera_distance
    camera_transform = numpy.dot(camera_transform, target_rotate)
    scene.set_view_matrix(numpy.linalg.inv(camera_transform))
    scene.set_projection(orthographic_projection)
    # render -------------------------------------------------------------------
    frame_buffer.enable()
    scene.mask_render(instances=target_names)
    instance_mask = frame_buffer.read_pixels()
    instance_depth_map = frame_buffer.read_pixels(
            read_depth=True, projection=orthographic_projection)
    
    #scene.color_render()
    #instance_color = frame_buffer.read_pixels()
    
    # restore the previous camera and hidden state =============================
    scene.set_view_matrix(original_view_matrix)
    scene.set_projection(original_projection)
    #for instance, hidden in hidden_instances.items():
    #    if hidden:
    #        scene.hide_instance(instance)
    #    else:
    #        scene.show_instance(instance)
    
    # check collision ==========================================================
    valid_pixels = numpy.sum(instance_mask != 0, axis=-1) != 0
    
    scene_depth_map -= camera_distance
    scene_depth_map *= -1.
    instance_depth_map -= camera_distance
    offset = (scene_depth_map - instance_depth_map).reshape(valid_pixels.shape)
    offset *= valid_pixels
    
    collision = numpy.max(offset) > max_intersection
    
    # dump images ==============================================================
    if dump_images is not None:
        save_image(scene_mask, './%s_scene_mask.png'%dump_images)
        save_image(instance_mask, './%s_instance_mask.png'%dump_images)
        save_depth(scene_depth_map, './%s_scene_depth.npy'%dump_images)
        save_depth(instance_depth_map, './%s_instance_depth.npy'%dump_images)
        #save_image(scene_color, './%s_scene_color.png'%dump_images)
        #save_image(instance_color, './%s_instance_color.png'%dump_images)
        
        collision_pixels = (offset > max_intersection).astype(numpy.uint8)
        collision_pixels = collision_pixels * 255
        save_image(collision_pixels, './%s_collision.png'%dump_images)
    
    return collision
