import { DashboardIcon, MagicWandIcon, MixerHorizontalIcon, BarChartIcon, CubeIcon, LayersIcon, GearIcon } from '@radix-ui/react-icons';
const ICONS = { overview: DashboardIcon, infer: MagicWandIcon, compare: MixerHorizontalIcon, metrics: BarChartIcon, models: CubeIcon, data: LayersIcon, about: GearIcon };
export default function NavigationIcon({ page }) {
  const Icon = ICONS[page];
  return <Icon aria-hidden="true" />;
}
