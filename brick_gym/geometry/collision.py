import numpy

from renderpy.frame_buffer import FrameBufferWrapper
from renderpy.camera import orthographic_matrix
from renderpy.image import save_image, save_depth

def check_collision(
        scene,
        target_instances,
        snap_transform,
        target_snap_gender,
        resolution=(64,64),
        frame_buffer=None,
        max_intersection=4,
        dump_images=None,
    ):
    
    #===========================================================================
    # setup
    #---------------------------------------------------------------------------
    # make sure the scene is renderable
    assert scene.renderable
    #---------------------------------------------------------------------------
    # build a renderpy frame buffer if a shared one was not specified
    if frame_buffer is None:
        frame_buffer = FrameBufferWrapper(
                resolution[0],
                resolution[1],
                anti_alias=False)
    #---------------------------------------------------------------------------
    # store the camera info and which bricks are hidden
    original_camera_pose = scene.get_camera_pose()
    original_camera_projection = scene.get_projection()
    hidden_instances = {i : scene.instance_hidden(i) for i in scene.instances}
    #---------------------------------------------------------------------------
    # compute the camera distance, clipping plane and the orthgraphic width
    camera_distance = 500 # TMP
    orthographic_width = 100 # TMP
    orthographic_height = 100 # TMP
    near_clip = 1 # TMP
    far_clip = 2000 # TMP
    
    #scene.set_ambient_color((1,1,1))
    scene.load_image_light('default', texture_directory='grey_cube')
    scene.set_active_image_light('default')
    
    #===========================================================================
    # render the scene depth map
    #---------------------------------------------------------------------------
    # show everything except for the target instances
    scene.show_all_instances()
    for instance in target_instances:
        scene.hide_instance(instance)
    #---------------------------------------------------------------------------
    # setup the camera
    camera_transform = snap_transform.copy()
    render_axis = snap_transform[:3,1]
    render_axis /= numpy.linalg.norm(render_axis)
    m_direction = -1
    f_direction = 1
    m_rotate = numpy.array([
            [1, 0, 0, 0],
            [0, 0,-1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]])
    f_rotate = numpy.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]])

    if target_snap_gender.upper() == 'M':
        scene_axis = render_axis * f_direction
        scene_rotate = f_rotate
        target_axis = render_axis * m_direction
        target_rotate = m_rotate
    else:
        scene_axis = render_axis * m_direction
        scene_rotate = m_rotate
        target_axis = render_axis * f_direction
        target_rotate = f_rotate
    camera_transform[:3,3] += scene_axis * camera_distance
    camera_transform = numpy.dot(camera_transform, scene_rotate)
    scene.set_camera_pose(numpy.linalg.inv(camera_transform))
    orthographic_projection = orthographic_matrix(
            l = -orthographic_width,
            r = orthographic_width,
            b = -orthographic_height,
            t = orthographic_height,
            n = near_clip,
            f = far_clip)
    scene.set_projection(orthographic_projection)
    #---------------------------------------------------------------------------
    # render
    frame_buffer.enable()
    scene.mask_render()
    scene_mask = frame_buffer.read_pixels()
    scene_depth_map = frame_buffer.read_pixels(
            read_depth=True, projection=orthographic_projection)
    
    scene.color_render()
    scene_color = frame_buffer.read_pixels()
    
    #===========================================================================
    # render the instance depth map
    #---------------------------------------------------------------------------
    # hide everything except for the target instance
    scene.hide_all_instances()
    for instance in target_instances:
        scene.show_instance(instance)
    #---------------------------------------------------------------------------
    # setup the camera
    camera_transform = snap_transform.copy()
    camera_transform[:3,3] += target_axis * camera_distance
    camera_transform = numpy.dot(camera_transform, target_rotate)
    scene.set_camera_pose(numpy.linalg.inv(camera_transform))
    scene.set_projection(orthographic_projection)
    #---------------------------------------------------------------------------
    # render
    frame_buffer.enable()
    scene.mask_render()
    instance_mask = frame_buffer.read_pixels()
    instance_depth_map = frame_buffer.read_pixels(
            read_depth=True, projection=orthographic_projection)
    
    scene.color_render()
    instance_color = frame_buffer.read_pixels()
    
    #===========================================================================
    # restore the previous camera and hidden state
    scene.set_camera_pose(original_camera_pose)
    scene.set_projection(original_camera_projection)
    for instance, hidden in hidden_instances.items():
        if hidden:
            scene.hide_instance(instance)
        else:
            scene.show_instance(instance)
    
    #===========================================================================
    # check collision
    valid_pixels = numpy.sum(instance_mask != 0, axis=-1) != 0
    
    scene_depth_map -= camera_distance
    scene_depth_map *= -1.
    instance_depth_map -= camera_distance
    offset = (scene_depth_map - instance_depth_map).reshape(valid_pixels.shape)
    offset *= valid_pixels
    
    collision = numpy.max(offset) > max_intersection
    
    #===========================================================================
    # dump images
    if dump_images is not None:
        save_image(scene_mask, './%s_scene_mask.png'%dump_images)
        save_image(instance_mask, './%s_instance_mask.png'%dump_images)
        save_depth(scene_depth_map, './%s_scene_depth.npy'%dump_images)
        save_depth(instance_depth_map, './%s_instance_depth.npy'%dump_images)
        save_image(scene_color, './%s_scene_color.png'%dump_images)
        save_image(instance_color, './%s_instance_color.png'%dump_images)
        
        collision_pixels = (offset > max_intersection).astype(numpy.uint8)
        collision_pixels = collision_pixels * 255
        save_image(collision_pixels, './%s_collision.png'%dump_images)
    
    return collision
