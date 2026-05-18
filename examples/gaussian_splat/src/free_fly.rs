use pill_engine::game::*;

pub struct FreeFlyComponent {
    pub yaw:         f32, // degrees, horizontal rotation around world Y
    pub pitch:       f32, // degrees, tilt up/down [-85, 85]
    pub speed:       f32,
    pub sensitivity: f32,
}

impl FreeFlyComponent {
    pub fn new(yaw: f32, pitch: f32, speed: f32) -> Self {
        Self { yaw, pitch, speed, sensitivity: 0.15 }
    }
}

impl PillTypeMapKey for FreeFlyComponent {
    type Storage = ComponentStorage<FreeFlyComponent>;
}

impl Component for FreeFlyComponent {}

pub fn free_fly_system(engine: &mut Engine) -> Result<()> {
    let dt = engine.get_global_component::<TimeComponent>()?.delta_time;

    let input       = engine.get_global_component::<InputComponent>()?;
    let look        = input.get_mouse_button(MouseButton::Left)
                   || input.get_mouse_button(MouseButton::Right);
    let mouse_delta = input.get_mouse_delta();
    let w           = input.get_key(KeyboardKey::KeyW);
    let s           = input.get_key(KeyboardKey::KeyS);
    let a           = input.get_key(KeyboardKey::KeyA);
    let d           = input.get_key(KeyboardKey::KeyD);
    let q           = input.get_key(KeyboardKey::KeyQ);
    let e           = input.get_key(KeyboardKey::KeyE);

    for (_, transform, fly) in
        engine.iterate_two_components_mut::<TransformComponent, FreeFlyComponent>()?
    {
        if look {
            fly.yaw   -= mouse_delta.x * fly.sensitivity;
            fly.pitch  = (fly.pitch - mouse_delta.y * fly.sensitivity).clamp(-85.0, 85.0);
        }

        let yaw_r   = fly.yaw.to_radians();
        let pitch_r = fly.pitch.to_radians();

        // Matches PassSplat convention: forward = yaw(Y) * pitch(P) * +Z
        let forward = Vector3f::new(
            yaw_r.sin() * pitch_r.cos(),
            -pitch_r.sin(),
            yaw_r.cos() * pitch_r.cos(),
        );
        // Strafe direction is purely horizontal (ignoring pitch), matching UE behaviour
        let right = Vector3f::new(yaw_r.cos(), 0.0, -yaw_r.sin());

        let mut vel = Vector3f::ZERO;
        if w { vel += forward; }
        if s { vel -= forward; }
        if d { vel += right; }
        if a { vel -= right; }
        if e { vel += Vector3f::Y; }
        if q { vel -= Vector3f::Y; }

        if vel.length_squared() > 0.0 {
            transform.set_position(transform.position + vel.normalize() * fly.speed * dt);
        }

        transform.set_rotation(Vector3f::new(fly.pitch, fly.yaw, 0.0));
    }

    Ok(())
}
