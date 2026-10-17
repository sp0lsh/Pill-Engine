//! Trace LODs: coarse copies of meshes for ray tracing only (the engine's GI BVH), where small
//! detail doesn't matter and every triangle costs on every ray. meshoptimizer's sloppy
//! simplification (it ignores topology) within an absolute error.

/// Bytes per RMSH vertex (`convert::CookedVertex`), position first.
const RMSH_VERTEX_BYTES: usize = 56;

/// The trace LOD of an RMSH mesh within `cell` (world units): positions and indices, empty when
/// the whole mesh is smaller than `cell` (not worth tracing). `None` when `rmsh` is malformed.
pub fn trace_lod(rmsh: &[u8], cell: f32) -> Option<(Vec<[f32; 3]>, Vec<u32>)> {
    let read_u32 = |offset: usize| {
        Some(u32::from_le_bytes(
            rmsh.get(offset..offset + 4)?.try_into().ok()?,
        ))
    };
    let (vertex_count, index_count) = (read_u32(8)? as usize, read_u32(12)? as usize);
    let vertices_end = 16 + vertex_count * RMSH_VERTEX_BYTES;
    let vertex_bytes = rmsh.get(16..vertices_end)?;
    let indices: Vec<u32> = rmsh
        .get(vertices_end..vertices_end + index_count * 4)?
        .chunks_exact(4)
        .map(|index| u32::from_le_bytes([index[0], index[1], index[2], index[3]]))
        .collect();
    let positions: Vec<[f32; 3]> = vertex_bytes
        .chunks_exact(RMSH_VERTEX_BYTES)
        .map(|vertex| {
            let coordinate = |i: usize| {
                f32::from_le_bytes([vertex[i], vertex[i + 1], vertex[i + 2], vertex[i + 3]])
            };
            [coordinate(0), coordinate(4), coordinate(8)]
        })
        .collect();
    let (min, max) = positions
        .iter()
        .fold(([f32::MAX; 3], [f32::MIN; 3]), |(low, high), p| {
            (
                [low[0].min(p[0]), low[1].min(p[1]), low[2].min(p[2])],
                [high[0].max(p[0]), high[1].max(p[1]), high[2].max(p[2])],
            )
        });
    if positions.is_empty() || (0..3).all(|axis| max[axis] - min[axis] < cell) {
        return Some((Vec::new(), Vec::new()));
    }
    let adapter = meshopt::VertexDataAdapter::new(vertex_bytes, RMSH_VERTEX_BYTES, 0).ok()?;
    // meshoptimizer's error is relative to the mesh's extent
    let scale = meshopt::simplify_scale(&adapter);
    let target_error = if scale > 0.0 { cell / scale } else { 1.0 };
    let simplified = meshopt::simplify_sloppy(&indices, &adapter, 0, target_error, None);
    // Only the vertices still used, renumbered
    let mut remap = vec![u32::MAX; positions.len()];
    let mut lod_positions = Vec::new();
    let lod_indices = simplified
        .iter()
        .map(|&index| {
            let slot = &mut remap[index as usize];
            if *slot == u32::MAX {
                *slot = lod_positions.len() as u32;
                lod_positions.push(positions[index as usize]);
            }
            *slot
        })
        .collect();
    Some((lod_positions, lod_indices))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::{rmsh_bytes, CookedVertex};

    // A `size` × `size` m plane of 1 / `per_metre` m quads
    fn plane(size: f32, per_metre: usize) -> Vec<u8> {
        let side = (size * per_metre as f32) as usize + 1;
        let step = 1.0 / per_metre as f32;
        let vertices: Vec<CookedVertex> = (0..side * side)
            .map(|i| CookedVertex {
                position: [(i % side) as f32 * step, 0.0, (i / side) as f32 * step],
                texture_coordinates: [0.0; 2],
                normal: [0.0, 1.0, 0.0],
                tangent: [1.0, 0.0, 0.0],
                bitangent: [0.0, 0.0, 1.0],
            })
            .collect();
        let mut indices = Vec::new();
        for z in 0..side - 1 {
            for x in 0..side - 1 {
                let corner = (z * side + x) as u32;
                let side = side as u32;
                indices.extend([
                    corner,
                    corner + side,
                    corner + 1,
                    corner + 1,
                    corner + side,
                    corner + side + 1,
                ]);
            }
        }
        rmsh_bytes(&vertices, &indices)
    }

    #[test]
    fn trace_lod_dense_plane_cell_quarter_metre_fewer_triangles() {
        let rmsh = plane(4.0, 20);
        let (positions, indices) = trace_lod(&rmsh, 0.25).unwrap();
        let full = 4 * 20 * 4 * 20 * 6;
        assert!(
            !indices.is_empty() && indices.len() < full / 4,
            "{} of {full}",
            indices.len()
        );
        assert!(indices
            .iter()
            .all(|&index| (index as usize) < positions.len()));
    }

    #[test]
    fn trace_lod_mesh_smaller_than_cell_empty() {
        let (positions, indices) = trace_lod(&plane(0.2, 20), 0.25).unwrap();
        assert!(positions.is_empty() && indices.is_empty());
    }
}
