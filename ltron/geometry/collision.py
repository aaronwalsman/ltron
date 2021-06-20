import numpy

from splendor.frame_buffer import FrameBufferWrapper
from splendor.camera import orthographic_matrix
from splendor.image import save_image, save_depth

def check_collision(
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
    hidden_instances = {i : scene.instance_hidden(i) for i in scene.instances}
    
    # compute the camera distance, clipping plane and the orthgraphic width ----
    #camera_distance = 500 # TMP
    #orthographic_width = 100 # TMP
    #orthographic_height = 100 # TMP
    #near_clip = 1 # TMP
    #far_clip = 2000 # TMP
    
    # render the scene depth map ===============================================
    # show everything except for the target instances --------------------------
    
    #scene.show_all_brick_instances()
    #scene.hide_all_snap_instances()
    #for instance in target_instances:
    #    scene.hide_instance(instance)
    
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
    else:
        scene_axis = render_axis * p_direction
        scene_rotate = p_rotate
        target_axis = render_axis * n_direction
        target_rotate = n_rotate
    
    # compute the relevant extents ---------------------------------------------
    local_vertices = []
    inv_snap_transform = numpy.linalg.inv(snap_transform)
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
            #n = near_clip,
            #f = far_clip)
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
