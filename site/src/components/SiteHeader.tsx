import {Link} from "@heroui/react";

import {isAgentsRoute, isFrameworksRoute} from "../lib/data";
import type {MirrorDatabase, Route} from "../types";
import {NewsTicker} from "./NewsTicker";

interface SiteHeaderProps {
  database: MirrorDatabase;
  route: Route;
  onNavigate: (hash: string) => void;
}

// caniuse's bar: Home and News left of the ticker, the sections and About right of it.
const navItems = [
  {label: "Home", href: "#/", route: "index"},
  {label: "News", href: "#/changes", route: "changes"},
  {label: "Frameworks", href: "#/frameworks", route: "frameworks"},
  {label: "Agents", href: "#/agents", route: "agents"},
  {label: "Stats", href: "#/stats", route: "stats"},
  {label: "About", href: "#/method", route: "method"},
] as const;

export function SiteHeader({route, onNavigate}: SiteHeaderProps) {
  return (
    <header className="site-header">
      <div className="site-shell header-main-row">
        <nav className="site-nav" aria-label="Primary navigation">
          <ul className="site-nav-list">
            {navItems.map((item) => {
              const active =
                route.name === item.route ||
                (item.route === "frameworks" && isFrameworksRoute(route)) ||
                (item.route === "agents" && isAgentsRoute(route));
              return (
                <li key={item.route}>
                  <Link
                    className="nav-link"
                    data-active={active || undefined}
                    href={item.href}
                    aria-current={active ? "page" : undefined}
                    aria-label={item.label}
                    onPress={() => onNavigate(item.href)}
                  >
                    {item.label}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>
        <div className="header-ticker-row">
          <NewsTicker />
        </div>
      </div>
    </header>
  );
}
