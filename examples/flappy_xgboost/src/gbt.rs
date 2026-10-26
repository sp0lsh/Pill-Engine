//! Gradient-boosted trees for binary logistic loss, the XGBoost recipe (Chen & Guestrin 2016):
//! second-order gradient statistics, L2-regularised leaf weights, exact greedy splits.

pub struct Params {
    pub trees: usize,
    pub depth: usize,
    /// Shrinkage applied to every leaf weight.
    pub eta: f32,
    /// L2 penalty on leaf weights.
    pub lambda: f32,
    /// Minimum gain a split must bring.
    pub gamma: f32,
    /// Minimum hessian sum on each side of a split.
    pub min_child_weight: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            trees: 40,
            depth: 3,
            eta: 0.3,
            lambda: 1.0,
            gamma: 0.0,
            min_child_weight: 1.0,
        }
    }
}

enum Node {
    /// Rows with `row[feature] < threshold` go left.
    Split {
        feature: usize,
        threshold: f32,
        left: usize,
        right: usize,
    },
    Leaf(f32),
}

pub struct Booster<const F: usize> {
    base_margin: f32,
    trees: Vec<Vec<Node>>,
}

pub fn sigmoid(margin: f32) -> f32 {
    1.0 / (1.0 + (-margin).exp())
}

fn tree_value<const F: usize>(tree: &[Node], row: &[f32; F]) -> f32 {
    let mut index = 0;
    loop {
        match tree[index] {
            Node::Split {
                feature,
                threshold,
                left,
                right,
            } => index = if row[feature] < threshold { left } else { right },
            Node::Leaf(weight) => return weight,
        }
    }
}

impl<const F: usize> Booster<F> {
    /// Probability of label 1.
    pub fn predict(&self, row: &[f32; F]) -> f32 {
        let margin = self.trees.iter().map(|tree| tree_value(tree, row)).sum::<f32>();
        sigmoid(self.base_margin + margin)
    }

    pub fn tree_count(&self) -> usize {
        self.trees.len()
    }

    /// Readable dump: the base margin, then one `tree` block per tree, one node per line.
    pub fn to_text(&self, names: &[&str; F]) -> String {
        let mut text = String::from(
            "# node lines: `<index> split <feature> <threshold> <left> <right>` or `<index> leaf <weight>`\n",
        );
        text.push_str(&format!("base_margin {}\n", self.base_margin));
        for (index, tree) in self.trees.iter().enumerate() {
            text.push_str(&format!("tree {index}\n"));
            for (node, value) in tree.iter().enumerate() {
                match *value {
                    Node::Split {
                        feature,
                        threshold,
                        left,
                        right,
                    } => text.push_str(&format!(
                        "{node} split {} {threshold} {left} {right}\n",
                        names[feature]
                    )),
                    Node::Leaf(weight) => text.push_str(&format!("{node} leaf {weight}\n")),
                }
            }
        }
        text
    }

    /// Parses `to_text` output; `None` when the text holds no model.
    pub fn from_text(text: &str, names: &[&str; F]) -> Option<Self> {
        let mut base_margin = None;
        let mut trees: Vec<Vec<Node>> = Vec::new();
        for line in text.lines().filter(|line| !line.starts_with('#')) {
            let words: Vec<&str> = line.split_whitespace().collect();
            match words.as_slice() {
                ["base_margin", value] => base_margin = Some(value.parse().ok()?),
                ["tree", _] => trees.push(Vec::new()),
                [_, "split", feature, threshold, left, right] => trees.last_mut()?.push(Node::Split {
                    feature: names.iter().position(|name| name == feature)?,
                    threshold: threshold.parse().ok()?,
                    left: left.parse().ok()?,
                    right: right.parse().ok()?,
                }),
                [_, "leaf", weight] => trees.last_mut()?.push(Node::Leaf(weight.parse().ok()?)),
                [] => {}
                _ => return None,
            }
        }
        Some(Self {
            base_margin: base_margin?,
            trees,
        })
    }
}

/// What one round of tree growth reads.
struct Round<'a, const F: usize> {
    rows: &'a [[f32; F]],
    /// Row indices sorted by each feature.
    sorted: &'a [Vec<usize>],
    gradients: &'a [f32],
    hessians: &'a [f32],
    params: &'a Params,
}

impl<const F: usize> Round<'_, F> {
    /// Grows the subtree of the rows in `member`, returns its node index.
    fn grow(&self, tree: &mut Vec<Node>, member: &[bool], depth: usize) -> usize {
        let lambda = self.params.lambda;
        let (mut gradient, mut hessian) = (0.0, 0.0);
        for (index, _) in member.iter().enumerate().filter(|(_, &inside)| inside) {
            gradient += self.gradients[index];
            hessian += self.hessians[index];
        }
        let node = tree.len();
        tree.push(Node::Leaf(-self.params.eta * gradient / (hessian + lambda)));
        if depth == self.params.depth {
            return node;
        }
        let score = |g: f32, h: f32| g * g / (h + lambda);
        let parent = score(gradient, hessian);
        // (gain, feature, threshold)
        let mut best: Option<(f32, usize, f32)> = None;
        for (feature, order) in self.sorted.iter().enumerate() {
            let (mut left_gradient, mut left_hessian) = (0.0, 0.0);
            let mut previous: Option<f32> = None;
            for &index in order.iter().filter(|&&index| member[index]) {
                let value = self.rows[index][feature];
                if let Some(previous) = previous.filter(|&previous| value > previous) {
                    let right_hessian = hessian - left_hessian;
                    if left_hessian >= self.params.min_child_weight
                        && right_hessian >= self.params.min_child_weight
                    {
                        let gain = 0.5
                            * (score(left_gradient, left_hessian)
                                + score(gradient - left_gradient, right_hessian)
                                - parent)
                            - self.params.gamma;
                        if gain > best.map_or(0.0, |(best_gain, _, _)| best_gain) {
                            best = Some((gain, feature, (previous + value) * 0.5));
                        }
                    }
                }
                left_gradient += self.gradients[index];
                left_hessian += self.hessians[index];
                previous = Some(value);
            }
        }
        let Some((_, feature, threshold)) = best else {
            return node;
        };
        let goes_left = |index: usize| self.rows[index][feature] < threshold;
        let left_member: Vec<bool> = (0..member.len()).map(|i| member[i] && goes_left(i)).collect();
        let right_member: Vec<bool> = (0..member.len()).map(|i| member[i] && !goes_left(i)).collect();
        let left = self.grow(tree, &left_member, depth + 1);
        let right = self.grow(tree, &right_member, depth + 1);
        tree[node] = Node::Split {
            feature,
            threshold,
            left,
            right,
        };
        node
    }
}

/// Fits `labels` (0 or 1) from `rows`.
pub fn fit<const F: usize>(rows: &[[f32; F]], labels: &[f32], params: &Params) -> Booster<F> {
    let count = rows.len();
    let mean = (labels.iter().sum::<f32>() / count.max(1) as f32).clamp(1e-3, 1.0 - 1e-3);
    let base_margin = (mean / (1.0 - mean)).ln();
    let sorted: Vec<Vec<usize>> = (0..F)
        .map(|feature| {
            let mut order: Vec<usize> = (0..count).collect();
            order.sort_by(|&a, &b| rows[a][feature].total_cmp(&rows[b][feature]));
            order
        })
        .collect();
    let mut margins = vec![base_margin; count];
    let mut gradients = vec![0.0; count];
    let mut hessians = vec![0.0; count];
    let everyone = vec![true; count];
    let mut trees = Vec::with_capacity(params.trees);
    for _ in 0..params.trees {
        for index in 0..count {
            let probability = sigmoid(margins[index]);
            gradients[index] = probability - labels[index];
            hessians[index] = (probability * (1.0 - probability)).max(1e-6);
        }
        let round = Round {
            rows,
            sorted: &sorted,
            gradients: &gradients,
            hessians: &hessians,
            params,
        };
        let mut tree = Vec::new();
        round.grow(&mut tree, &everyone, 0);
        for (margin, row) in margins.iter_mut().zip(rows) {
            *margin += tree_value(&tree, row);
        }
        trees.push(tree);
    }
    Booster {
        base_margin,
        trees,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_threshold_dataset_separates_classes() {
        let rows: Vec<[f32; 2]> = (0..100).map(|i| [i as f32, (i % 7) as f32]).collect();
        let labels: Vec<f32> = (0..100).map(|i| if i >= 50 { 1.0 } else { 0.0 }).collect();
        let booster = fit(&rows, &labels, &Params::default());
        assert!(booster.predict(&[10.0, 3.0]) < 0.1);
        assert!(booster.predict(&[90.0, 3.0]) > 0.9);
    }

    #[test]
    fn fit_xor_dataset_depth_2_accuracy_1() {
        let mut rows = Vec::new();
        let mut labels = Vec::new();
        // Balanced xor gives every root split zero gain; one doubled quadrant breaks the tie
        for (a, b, copies) in [(0.0, 0.0, 50), (0.0, 1.0, 50), (1.0, 0.0, 50), (1.0, 1.0, 100)] {
            for _ in 0..copies {
                rows.push([a, b]);
                labels.push(if a != b { 1.0 } else { 0.0 });
            }
        }
        let params = Params {
            depth: 2,
            ..Params::default()
        };
        let booster = fit(&rows, &labels, &params);
        for (row, label) in rows.iter().zip(&labels) {
            assert_eq!(booster.predict(row) > 0.5, *label > 0.5);
        }
    }

    #[test]
    fn to_text_from_text_round_trip_predicts_the_same() {
        let rows: Vec<[f32; 2]> = (0..100).map(|i| [i as f32, (i % 7) as f32]).collect();
        let labels: Vec<f32> = (0..100).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
        let names = ["a", "b"];
        let booster = fit(&rows, &labels, &Params::default());
        let parsed = Booster::from_text(&booster.to_text(&names), &names).expect("parses");
        for row in &rows {
            assert_eq!(booster.predict(row), parsed.predict(row));
        }
    }

    #[test]
    fn from_text_empty_returns_none() {
        assert!(Booster::<2>::from_text("", &["a", "b"]).is_none());
    }

    #[test]
    fn predict_zero_trees_returns_sigmoid_of_base_margin() {
        let rows = [[0.0], [1.0], [2.0], [3.0]];
        let labels = [0.0, 1.0, 1.0, 1.0];
        let params = Params {
            trees: 0,
            ..Params::default()
        };
        let booster = fit(&rows, &labels, &params);
        assert!((booster.predict(&[0.0]) - 0.75).abs() < 1e-5);
    }
}
