//! Shader bindings generated at build time from slangc's reflection of res/shaders/*.hlsl
//! (pill_assets::shader_bindings): per shader a module (named like the shader file) with
//! `layout()` (bind group layout entries per group), a `Binding` const per resource and the
//! override ids of its specialization constants; `structs` holds the `#[repr(C)]` mirror of every
//! uniform / storage struct.

include!(concat!(env!("OUT_DIR"), "/shader_bindings.rs"));

/// Bind group and binding index of one shader resource.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Binding {
    pub(crate) group: u32,
    pub(crate) binding: u32,
}

/// Pipeline layout entries of shaders used together (vertex + fragment): the union of their
/// groups, with the stage visibility of a binding both declare combined.
#[allow(dead_code)]
pub(crate) fn merge_layouts(
    layouts: &[Vec<Vec<wgpu::BindGroupLayoutEntry>>],
) -> Vec<Vec<wgpu::BindGroupLayoutEntry>> {
    let group_count = layouts.iter().map(Vec::len).max().unwrap_or(0);
    let mut merged: Vec<Vec<wgpu::BindGroupLayoutEntry>> = vec![Vec::new(); group_count];
    for layout in layouts {
        for (group, entries) in layout.iter().enumerate() {
            for entry in entries {
                match merged[group]
                    .iter_mut()
                    .find(|existing| existing.binding == entry.binding)
                {
                    Some(existing) => {
                        assert_eq!(
                            existing.ty, entry.ty,
                            "group {group} binding {} differs between shaders",
                            entry.binding
                        );
                        existing.visibility |= entry.visibility;
                    }
                    None => merged[group].push(*entry),
                }
            }
        }
    }
    for entries in &mut merged {
        entries.sort_by_key(|entry| entry.binding);
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    #[test]
    fn merge_shared_binding_ors_visibility_and_keeps_others() {
        let vertex = vec![vec![uniform(0, wgpu::ShaderStages::VERTEX)]];
        let fragment = vec![
            vec![
                uniform(1, wgpu::ShaderStages::FRAGMENT),
                uniform(0, wgpu::ShaderStages::FRAGMENT),
            ],
            vec![uniform(0, wgpu::ShaderStages::FRAGMENT)],
        ];
        let merged = merge_layouts(&[vertex, fragment]);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0][0].visibility, wgpu::ShaderStages::VERTEX_FRAGMENT);
        assert_eq!(merged[0][1].binding, 1);
        assert_eq!(merged[1].len(), 1);
    }

    #[test]
    fn generated_tonemap_layout_matches_shader_declarations() {
        let layout = tonemap_fragment::layout();
        assert_eq!(layout.len(), 1);
        assert_eq!(tonemap_fragment::PARAMS.binding, 2);
        assert_eq!(std::mem::size_of::<structs::TonemapParams>(), 32);
    }
}
