//! The few plain HTTP GETs the monitor makes to a server's operator endpoints.
use std::{
    io::{Read as _, Write as _},
    net::{SocketAddr, TcpStream, ToSocketAddrs as _},
    time::Duration,
};
use tur::{Result, TurError};

const TIMEOUT: Duration = Duration::from_secs(2);

/// Where a server listens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Server {
    host: String,
}

impl Server {
    /// Accepts `http://host:port` or `host:port`. Tur serves plain HTTP only.
    pub(crate) fn parse(url: &str) -> Result<Self> {
        let host = url.strip_prefix("http://").unwrap_or(url);
        let host = host.trim_end_matches('/');
        if host.is_empty() || host.contains('/') || host.contains("://") {
            return Err(TurError::Other(format!(
                "server URL must look like http://host:port, not {url}"
            )));
        }
        Ok(Self { host: host.into() })
    }

    /// The address a local client reaches a server on when it binds `listen`.
    pub(crate) fn listening_on(listen: SocketAddr) -> Self {
        let mut reachable = listen;
        if listen.ip().is_unspecified() {
            reachable.set_ip(std::net::Ipv4Addr::LOCALHOST.into());
        }
        Self {
            host: reachable.to_string(),
        }
    }

    /// Whether every address the host resolves to is on this machine.
    pub(crate) fn is_local(&self) -> Result<bool> {
        let mut addresses = self.host.to_socket_addrs()?.peekable();
        Ok(addresses.peek().is_some() && addresses.all(|address| address.ip().is_loopback()))
    }

    pub(crate) fn url(&self) -> String {
        format!("http://{}", self.host)
    }

    /// Fetch `path` and return its status and body.
    pub(crate) fn get(&self, path: &str) -> Result<(u16, String)> {
        let unreachable = |error: &dyn std::fmt::Display| {
            TurError::Other(format!("no Tur server at {}: {error}", self.url()))
        };
        let address = self
            .host
            .to_socket_addrs()
            .map_err(|error| unreachable(&error))?
            .next()
            .ok_or_else(|| unreachable(&"the name has no address"))?;
        let mut stream =
            TcpStream::connect_timeout(&address, TIMEOUT).map_err(|error| unreachable(&error))?;
        stream.set_read_timeout(Some(TIMEOUT))?;
        stream.set_write_timeout(Some(TIMEOUT))?;
        write!(
            stream,
            "GET {path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
            self.host
        )?;
        let mut response = String::new();
        stream.read_to_string(&mut response)?;
        parse_response(&response)
    }
}

fn parse_response(response: &str) -> Result<(u16, String)> {
    let malformed = || TurError::Other("the server sent a malformed HTTP response".into());
    let (head, body) = response.split_once("\r\n\r\n").ok_or_else(malformed)?;
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|status| status.parse().ok())
        .ok_or_else(malformed)?;
    Ok((status, body.to_owned()))
}

#[cfg(test)]
#[allow(clippy::panic_in_result_fn, reason = "tests assert parsing outcomes")]
mod tests {
    use super::*;

    #[test]
    fn test_server_urls_accept_a_scheme_and_reject_other_shapes() -> Result<()> {
        assert_eq!(Server::parse("http://a:1/")?.url(), "http://a:1");
        assert_eq!(Server::parse("a:1")?.url(), "http://a:1");
        assert!(Server::parse("https://a:1").is_err());
        assert!(Server::parse("a:1/v1").is_err());
        Ok(())
    }

    #[test]
    fn test_wildcard_listener_is_reached_on_loopback() {
        let server = Server::listening_on(SocketAddr::from(([0, 0, 0, 0], 8080)));
        assert_eq!(server.url(), "http://127.0.0.1:8080");
    }

    #[test]
    fn test_only_loopback_addresses_count_as_local() -> Result<()> {
        assert!(Server::parse("127.0.0.1:8080")?.is_local()?);
        assert!(Server::parse("[::1]:8080")?.is_local()?);
        assert!(!Server::parse("192.0.2.1:8080")?.is_local()?);
        Ok(())
    }

    #[test]
    fn test_response_splits_status_from_body() -> Result<()> {
        let (status, body) = parse_response("HTTP/1.1 503 X\r\nA: b\r\n\r\n{}")?;
        assert_eq!((status, body.as_str()), (503, "{}"));
        assert!(parse_response("nonsense").is_err());
        Ok(())
    }
}
