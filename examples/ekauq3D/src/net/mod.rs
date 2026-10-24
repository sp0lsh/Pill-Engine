//! Network play (native): a listen server in the hosting game (`EKAUQ_HOST`), clients joining it
//! (`EKAUQ_CONNECT`), the dedicated server (`examples/ekauq_server.rs`); game-level messages over
//! the engine's renet (UDP) transport.

pub mod client;
pub mod protocol;
pub mod server;

#[cfg(test)]
mod tests {
    use super::{client::Connection, server::Host};
    use crate::{
        bots::{
            calibrate::calibrate,
            genome::{Genome, Knowledge},
            nav::NavGraph,
            Bots,
        },
        sim::{
            arena::{box_world, Arena, Spawn},
            combat::{Command, Event},
            session::Session,
            weapons::{hash, WEAPONS},
            STEP,
        },
    };

    /// A server and two clients over UDP on this machine: both join, their commands move their
    /// fighters on the server, both see the server's events and each other in snapshots.
    #[test]
    fn loopback_server_and_two_clients() {
        let collision = box_world(&[([-4096, -4096, -64], [4096, 4096, 0])]);
        let z = 24.0 + 1.0 / 32.0;
        let spawns = [[-512.0, 0.0], [512.0, 0.0], [0.0, 512.0], [0.0, -512.0]]
            .iter()
            .map(|&[x, y]| Spawn {
                origin: [x, y, z],
                yaw: 0.0,
            })
            .collect();
        let arena = Arena::from_points(spawns, Vec::new(), Vec::new());
        let knowledge = Knowledge {
            weapons: hash(&WEAPONS),
            calibration: calibrate(&WEAPONS, 1),
            genome: Genome::default(),
        };
        let mut session = Session::new(&arena, 3, 1, 0.5, 4);
        let nav = NavGraph::of(&collision, &arena);
        let mut bots = Bots::new(&session, &arena, nav, knowledge);
        let port = 27611;
        let mut host = Host::start(port, "test").unwrap();
        let address = format!("127.0.0.1:{port}");
        let mut clients = [
            Connection::connect(&address, "test").unwrap(),
            Connection::connect(&address, "test").unwrap(),
        ];
        let mut events = [Vec::<Event>::new(), Vec::new()];
        let tick = std::time::Duration::from_secs_f32(STEP);
        for step in 0..(4.0 / STEP) as usize {
            host.receive(tick, &mut session, &arena);
            let players = host.take_commands();
            let commands = bots.commands(&session, &collision, &arena, &players);
            let (happened, _) = session.advance(&collision, &arena, &commands);
            host.send_tick(&session, &happened);
            host.flush();
            for (index, client) in clients.iter_mut().enumerate() {
                client.receive(STEP, &collision);
                events[index].append(&mut client.events);
                if client.slot().is_some() {
                    // The first client runs along +y after a second
                    let mut command = Command::default();
                    command.input.yaw = 90.0;
                    command.input.forward = if index == 0 && step > 72 { 1.0 } else { 0.0 };
                    client.send_command(command, &collision);
                }
                client.flush();
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let mut slots: Vec<usize> = clients.iter().filter_map(Connection::slot).collect();
        let (runner_slot, stander_slot) = (slots[0], slots[1]);
        slots.sort_unstable();
        assert_eq!(slots, vec![1, 2]);
        assert_eq!(host.players(), 2);
        let runner = &session.fight.fighters[runner_slot];
        let stander = &session.fight.fighters[stander_slot];
        assert!(
            runner.state.velocity[1] > 300.0,
            "runner {:?}",
            runner.state.velocity
        );
        assert!(stander.state.velocity[1].abs() < 1.0);
        // Each client heard of its own arrival in the match
        for (client, seen) in clients.iter().zip(&events) {
            let slot = client.slot().unwrap();
            assert!(
                seen.iter()
                    .any(|event| matches!(event, Event::Respawned { fighter } if *fighter == slot)),
                "slot {slot} events {seen:?}"
            );
        }
        // The runner's own prediction is where the server has it (a snapshot behind at most)
        let predicted = clients[0].predicted.origin;
        let gap = crate::sim::distance(predicted, runner.state.origin);
        assert!(
            gap < 320.0 * 0.15,
            "predicted {predicted:?} server {:?}",
            runner.state.origin
        );
    }
}
